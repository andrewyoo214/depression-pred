# Depression-prediction

Major Depression Disorder(MDD) prediction with patients' bio-signal dataset

Analysis based on HRV(Heart Rate Variability), Bio-feedback, EEG(Electroencephalogram), ECG(Electrocardiogram), Sleep Track, etc.

### 1. Dataset codebook

* sub: 환자번호 (E001~E108까지 총 108명)
* session: 방문차수 (1에서 5까지)
* disorder: 질병 {1-Depression(주요우울장애), 2-Panic(공황장애), 3-Control(정상대조군)}
* age: 나이
* gender: 성별
* HAMD: 우울증 평가 점수 *
* HAMA: 불안증 평가 점수 *
* PDSS: 공황장애 점수 *
* ASI: 불안 민감성 점수 *
* APPQ: 공황 공포 점수 *
* PSWQ: 걱정 *
* SPI: 스트레스 *
* PSS: 스트레스 *
* BIS: 충동성 *
* SSI: 자살생각 *
* sDNN: Standard Deviation of Normal to Normal interval
* NN50: Number of pairs of adjacent NN intervals differing by more than 50 ms in the entire recording
* PNN50: NN50 count divided by the total number of all NN interval
* RMSSD: Root Mean Square Differences of Successive R-R intervals
* VLF: Very Low Frequency
* LF: Low Frequency
* HF: High Frequency
* LF/HF: 저주파, 고주파 비율 (교감과 부교감 신경의 균형 정도를 반영)
* Power: 주파수 영역의 강도
* HR: Heart Rate (심박수)
* Resp: Respiration(호흡)
* SC: 
* Temp: Temperature (체온)
* b: baseline status (b1, b2, b3)
* s: stress status 
* r: recovery status
