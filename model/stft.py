import torch.nn as nn
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

class AudioPreprocessor(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super(AudioPreprocessor, self).__init__()

        # STFT 계산 시 사용할 창 함수(window) 설정
        window = 'hann' # 일반적으로 사용되는 Hann window. 시간 영역에서 신호의 변화를 부드럽게 처리
        # STFT 계산 시 프레임 중심 정렬 여부
        center = True # True로 설정하면 입력 신호를 가운데 정렬하여 zero-padding 처리
        # STFT 계산 시 사용할 padding 모드
        pad_mode = 'reflect' # 경계에서 값을 반사(reflection)하여 패딩
        # Log-mel 스펙트로그램 계산에 사용할 기준(reference)과 하한값
        ref = 1.0
        amin = 1e-10 # 로그 스케일 계산 시 log(0) 방지를 위한 최소값
        top_db = None # None이면 모든 dynamic range 유지

        # 입력된 1차원 오디오 신호에 대해 STFT를 수행하여 
        # 복소수 형태의 스펙트로그램을 생성하고, 매그니튜드만 반환함.
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Spectrogram 결과를 mel 필터뱅크로 변환하고 로그 스케일을 적용함.
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # 오버피팅을 방지하기 위해 특정 시간/주파수 영역을 랜덤하게 드롭함.
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

    def forward(self, input, training=False):
        # 입력 오디오(1D 시퀀스)에 대해 STFT 수행 → 복소수 magnitude 스펙트럼 반환
        x = self.spectrogram_extractor(input)
        # STFT 결과를 mel 스케일로 변환한 후, 로그 스케일 적용
        x = self.logmel_extractor(x)
        
        # 학습 모드일 때만 SpecAugmentation 적용
        if training:
            # print(">>> SpecAugmentation 적용됨!")
            x = self.spec_augmenter(x)

        return x