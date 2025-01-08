mkdir llm-finetuner
cd llm-finetuner
mkdir backend frontend
cd frontend
npx create-react-app .
cd ../backend
pip install flask flask-cors torch transformers 