// ---------------------------
// 1️⃣ 학습 데이터 기반 Min-Max 범위
const mins = [0.0, 0.0, 0.0, 0.0];   
const maxs = [0.1, 0.2, 150.0, 80.0]; 

// ---------------------------
// 2️⃣ 모델 로드
let model;
async function loadModel() {
  model = await tf.loadLayersModel('./model/tfjs_model/model.json');
  console.log("모델 로드 완료");
} 

// ---------------------------
// 3️⃣ 임시 데이터 24시간 × 4 feature
async function fetchAirData() {
  const tempData = [
    [0.02, 0.05, 30, 12],
    [0.03, 0.06, 28, 15],
    [0.02, 0.04, 35, 10],
    [0.03, 0.05, 32, 14],
    [0.01, 0.04, 29, 11],
    [0.02, 0.05, 31, 13],
    [0.03, 0.06, 33, 15],
    [0.02, 0.05, 30, 12],
    [0.03, 0.06, 28, 15],
    [0.02, 0.04, 35, 10],
    [0.03, 0.05, 32, 14],
    [0.01, 0.04, 29, 11],
    [0.02, 0.05, 31, 13],
    [0.03, 0.06, 33, 15],
    [0.02, 0.05, 30, 12],
    [0.03, 0.06, 28, 15],
    [0.02, 0.04, 35, 10],
    [0.03, 0.05, 32, 14],
    [0.01, 0.04, 29, 11],
    [0.02, 0.05, 31, 13],
    [0.03, 0.06, 33, 15],
    [0.02, 0.05, 30, 12],
    [0.03, 0.06, 28, 15],
    [0.02, 0.04, 35, 10]
  ];
  return tempData;
}

// ---------------------------
// 4️⃣ Min-Max 정규화
function normalizeInput(input) {
  return input.map(row => row.map((x, i) => (x - mins[i]) / (maxs[i] - mins[i])));
}

// ---------------------------
// 5️⃣ 예측 함수
async function predictAirQuality() {
  if (!model) await loadModel();

  const rawData = await fetchAirData(); // [24,4]
  const inputData = normalizeInput(rawData);

  // tensor shape: [1,24,4], dtype float32 명시
  const tensor = tf.tensor3d(inputData, [24,4], 'float32'); // shape [24,4]
  const tensorBatch = tensor.expandDims(0);                 // shape [1,24,4]
  const output = model.predict(tensor);

  const pred = output.arraySync()[0]; // [2] 형태
  console.log("예측값:", pred);

  const resultEl = document.getElementById("result");
  if(resultEl) {
    resultEl.innerText = `미세먼지: ${pred[0].toFixed(2)}, 초미세먼지: ${pred[1].toFixed(2)}`;
  }
}

// ---------------------------
// 6️⃣ 버튼 클릭 이벤트 연결 (DOMContentLoaded 안에서)
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("predictBtn");
  if(btn) btn.addEventListener("click", predictAirQuality);
});

// 챗지피티 코드