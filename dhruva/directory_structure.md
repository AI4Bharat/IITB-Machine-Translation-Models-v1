./dhruva
├── Dockerfile
├── README.md
├── azure_ml
│   ├── README.md
│   ├── deployment.yml
│   ├── endpoint.yml
│   ├── environment.yml
│   └── model.yml
├── checkpoints
│   ├── en-hi
│   │   └── v1.0
│   │       ├── bpe-codes
│   │       │   ├── codes.en
│   │       │   └── codes.hi
│   │       └── model_ct2
│   │           ├── model.bin
│   │           ├── source_vocabulary.txt
│   │           └── target_vocabulary.txt
│   ├── en-mr
│   │   └── v1.0
│   │       ├── bpe-codes
│   │       │   ├── codes.en
│   │       │   └── codes.mr
│   │       └── model_ct2
│   │           ├── model.bin
│   │           ├── source_vocabulary.txt
│   │           └── target_vocabulary.txt
│   ├── hi-en
│   │   └── v1.0
│   │       ├── bpe-codes
│   │       │   ├── codes.en
│   │       │   └── codes.hi
│   │       └── model_ct2
│   │           ├── model.bin
│   │           ├── source_vocabulary.txt
│   │           └── target_vocabulary.txt
│   ├── hi-mr
│   │   └── v1.0
│   │       ├── bpe-codes
│   │       │   ├── codes.hi
│   │       │   └── codes.mr
│   │       └── model_ct2
│   │           ├── model.bin
│   │           ├── source_vocabulary.txt
│   │           └── target_vocabulary.txt
│   ├── mr-en
│   │   └── v1.0
│   │       ├── bpe-codes
│   │       │   ├── codes.en
│   │       │   └── codes.mr
│   │       └── model_ct2
│   │           ├── model.bin
│   │           ├── source_vocabulary.txt
│   │           └── target_vocabulary.txt
│   └── mr-hi
│       └── v1.0
│           ├── bpe-codes
│           │   ├── codes.hi
│           │   └── codes.mr
│           └── model_ct2
│               ├── model.bin
│               ├── source_vocabulary.txt
│               └── target_vocabulary.txt
├── client.py
├── directory_structure.md
├── requirements.txt
└── triton_repo
    └── nmt
        ├── 1
        │   ├── engine.py
        │   └── model.py
        └── config.pbtxt
