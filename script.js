// 모델과 설정 파일 로드
let model = null;
let scalerParams = null;

async function loadModel() {
    try {
        // 경로 앞에 model/ 추가
        model = await tf.loadLayersModel('./model/tfjs_model/model.json');
        model.build([null, 24, 4]);  // input shape 명시적 설정
        
        const configResponse = await fetch('./model/model_config.json');
        const config = await configResponse.json();
        scalerParams = config.scaler_params;
        
        console.log('모델 로드 완료');
    } catch (error) {
        console.error('모델 로딩 실패:', error);
    }
}

window.addEventListener('load', loadModel);