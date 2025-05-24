import torch
import torch.nn as nn
import torch.nn.functional as F

# 선형 및 합성곱 계층 초기화 함수
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)

# 배치 정규화 계층 초기화 함수
def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

# Mixup 데이터 증강 함수
def do_mixup(x, mixup_lambda):
    """
    Mixup: 짝수-홀수 인덱스의 데이터를 섞어주는 증강 기법
    x: (batch_size * 2, ...)
    mixup_lambda: (batch_size * 2,)
    """
    out = (x[0::2].transpose(0, -1) * mixup_lambda[0::2] +
           x[1::2].transpose(0, -1) * mixup_lambda[1::2]).transpose(0, -1)
    return out

# 기본적인 두 겹의 합성곱 블록 (Conv-BN-ReLU x 2 + Pooling)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    # 모든 계층 초기화
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    # Conv-BN-ReLU → Conv-BN-ReLU
    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))

        # 풀링 타입 선택
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise ValueError("Invalid pool_type.")
        
        return x

# PANNs CNN14 Model
class Cnn14(nn.Module):
    def __init__(self, classes_num):
        """
        입력: (batch_size, 1, time_steps, mel_bins)
        출력: clipwise_output (멀티클래스 확률), embedding (2048-d 벡터)
        """
        super(Cnn14, self).__init__()

        # 입력 전처리용 BatchNorm (1 → 64 채널에 대응)
        self.bn0 = nn.BatchNorm2d(64)

        # 점점 깊어지는 ConvBlock 계층 (특징 추출기)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        # 완전연결 계층
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        # BN 및 FC 계층 초기화
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x, mixup_lambda=None):
        """
        입력: log-mel 스펙트로그램 (batch_size, 1, time_steps, mel_bins)
        """
        x = x.transpose(1, 3) # (B, 1, T, mel_bins) → (B, mel_bins, T, 1)
        x = self.bn0(x) # 초기 BN 적용 (채널 수 64와 맞추기 위해)
        x = x.transpose(1, 3) # 다시 (B, 1, T, mel_bins) 형태로 복원

        # Mixup 적용 (훈련 중일 때만)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        # 합성곱 + 드롭아웃
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, 0.2, self.training)

        # 주파수 차원 평균 (B, 2048, time_steps)
        x = torch.mean(x, dim=3)

        # 시간 축에서 max pooling + average pooling 후 더함
        (x1, _) = torch.max(x, dim=2) # (B, 2048)
        x2 = torch.mean(x, dim=2) # (B, 2048)
        x = x1 + x2 # (B, 2048)

        # FC 계층을 통한 임베딩 추출 및 출력 예측
        x = F.dropout(x, 0.5, self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, 0.5, self.training) # 임베딩 출력 (fine-tuning에 사용 가능)
        clipwise_output = torch.sigmoid(self.fc_audioset(x)) # 각 클래스에 대한 확률 (multi-label)

        output_dict = {'clipwise_output': clipwise_output, # 예측 확률 출력
                       'embedding': embedding} # 특성 벡터 (2048차원)
        
        return output_dict