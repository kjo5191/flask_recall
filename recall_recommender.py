from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # ✅ 추가: 모든 React 요청 허용

# 데이터 불러오기
df = pd.read_csv('recall.csv')

# 추천용 문서 만들기
def build_document(row):
	return f"{row['manufacturer']} {row['model_name']} {row['recall_type']} {row['additional_info']}"

documents = df.apply(build_document, axis=1)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
print(f"[INFO] 전체 문서에서 추출된 고유 단어 수 (벡터 차원 수): {len(vectorizer.get_feature_names_out())}")

@app.route('/recommend', methods=['GET'])
def recommend():
	target_id = int(request.args.get('id'))
	if target_id not in df['id'].values:
		return jsonify([])

	idx = df.index[df['id'] == target_id][0]
	target_vec = tfidf_matrix[idx]
	similarities = cosine_similarity(target_vec, tfidf_matrix).flatten()

	# 자기 자신 제외
	similarities[idx] = -1

	top_indices = similarities.argsort()[::-1][:5]
	result_ids = df.iloc[top_indices]['id'].tolist()
	return jsonify(result_ids)

# if __name__ == '__main__':
# 	app.run(debug=True, port=5000)
# 아래는 배포버전
if __name__ == '__main__':
	import os
	port = int(os.environ.get('PORT', 10000))
	app.run(debug=False, host='0.0.0.0', port=port)

