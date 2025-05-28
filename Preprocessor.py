import torch
from torchaudio.transforms import MelScale, FrequencyMasking, TimeMasking

def logmel(waveform, sample_rate=16000, n_fft=512, hop_length=160, win_length=512, n_mels=64, fmin=50, fmax=8000, top_db=None):
    """
    waveform: [channel, num_samples] 또는 [num_samples]
    return: [channel, mel_bins, time]
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    window = torch.hann_window(win_length, device=waveform.device)

    # STFT
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True, pad_mode="reflect", return_complex=True)
    magnitude = torch.abs(stft)
    
    # 멜 스펙트로그램
    melfilter = MelScale(n_mels=n_mels, sample_rate=sample_rate, f_min=fmin, f_max=fmax, n_stft=magnitude.shape[1], norm=None, mel_scale="htk").to(magnitude.device)
    mel = melfilter(magnitude)

    # 로그 스케일
    logmel = torch.log(torch.clamp(mel, min=1e-10))

    # 다이나믹 레인지
    if top_db is not None:
        logmel = torch.maximum(logmel, logmel.max(dim=-1, keepdim=True).values - top_db)

    return logmel


def spec_augmentation(logmel, time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2):
    """
    logmel: [channel, mel_bins, time]
    - time_drop_width: 시간 마스킹 폭
    - time_stripes_num: 시간 마스킹 반복 횟수
    - freq_drop_width: 주파수 마스킹 폭
    - freq_stripes_num: 주파수 마스킹 반복 횟수
    """
    freq_mask = FrequencyMasking(freq_mask_param=freq_drop_width)
    time_mask = TimeMasking(time_mask_param=time_drop_width)

    for _ in range(freq_stripes_num):
        logmel_aug = freq_mask(logmel)

    for _ in range(time_stripes_num):
        logmel_aug = time_mask(logmel)
    
    return logmel_aug