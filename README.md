# mussel-assessment
A computer vision-based quality assessment model of Green mussel (perna viridis)

1. ## Prerequisites

Make sure you have these installed before proceeding:

- [Git](https://git-scm.com/downloads)
- [Python 3.10](https://www.python.org/downloads/)
- [Node.js](https://nodejs.org/) (v18 or higher)


## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/porxfzio/mussel-assessment.git
cd mussel-assessment
```

### 2. Download the model weights
Download the weights folder from Google Drive:
Place the files in `backend/weights/` locally
```bash
backend/weights/model.pth
backend/weights/rf_stage1_shell.pkl
backend/weights/rf_stage2_final.pkl
backend/weights/label_encoder.pkl
```
### 3. Create the .env file inside backend/
```bash
SUPABASE_URL=https://********.supabase.co
SUPABASE_SERVICE_KEY=********
```
### 4. Run backend in terminal or cmd
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
### 5. Run frontend (another terminal or cmd)
```bash
cd frontend
npm install
npm run dev
```

## Folder structure
```bash
mussel-assessment/
├── backend/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── maskrcnn.py
│   │   └── classifier.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── session.py
│   │   └── predict.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   └── supabase.py
│   ├── weights/
│   │   ├── README.txt          ← download instructions
│   │   ├── model.pth           ← not in GitHub
│   │   ├── rf_stage1_shell.pkl ← not in GitHub
│   │   ├── rf_stage2_final.pkl ← not in GitHub
│   │   └── label_encoder.pkl   ← not in GitHub
│   ├── .env                    ← not in GitHub
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── style.css
│   ├── public/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── .gitignore
└── README.md
```

