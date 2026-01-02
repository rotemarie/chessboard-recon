# Chessboard Recognition Dataset

## 📥 Download Instructions

The raw data for this project is hosted externally due to its size (~3-6 GB).

### Required Data Files

You need to download and place the following in this directory:

```
data/
├── per_frame/              # Labeled games (REQUIRED for training)
│   ├── game2_per_frame/
│   ├── game4_per_frame/
│   ├── game5_per_frame/
│   ├── game6_per_frame/
│   └── game7_per_frame/
└── PGN/                    # Additional unlabeled games (OPTIONAL)
    ├── c06/
    └── c17/
```

### Download Link

**📦 Download the data here:**
- **Link:** [ADD YOUR SHARED DRIVE LINK HERE]
- **Size:** ~X GB
- **Format:** ZIP file

### Setup Instructions

1. **Download the ZIP file** from the link above

2. **Extract to the correct location:**
   ```bash
   # If you downloaded to ~/Downloads/chessboard-data.zip
   cd "/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon"
   
   # Extract directly into data/
   unzip ~/Downloads/chessboard-data.zip -d data/
   ```

3. **Verify the structure:**
   ```bash
   ls data/per_frame/
   # Should see: game2_per_frame  game4_per_frame  game5_per_frame  game6_per_frame  game7_per_frame
   ```

4. **Continue with preprocessing:**
   ```bash
   cd preprocessing
   python preprocess_data.py
   ```

## 📊 Dataset Statistics

### Labeled Data (per_frame)
- **game2**: 77 labeled frames
- **game4**: 184 labeled frames
- **game5**: 109 labeled frames
- **game6**: 92 labeled frames
- **game7**: 55 labeled frames
- **Total**: ~517 frames → ~33,088 labeled squares

### Unlabeled Data (PGN) - Optional
- **game8-10**: ~27,534 frames
- **game11-13**: ~38,771 frames
- **Total**: ~66,305 additional frames

## 🔄 Alternative: Generate Preprocessed Data

If you only need the preprocessed data for training (not the raw images):

**Option A:** Download preprocessed data only
- **Link:** [ADD PREPROCESSED DATA LINK HERE]
- **Size:** ~500 MB (much smaller!)
- Extract to: `preprocessed_data/`

**Option B:** Generate it yourself
```bash
# If you have the raw data
cd preprocessing
python preprocess_data.py
# Takes 5-10 minutes, outputs to preprocessed_data/
```

## ❓ Troubleshooting

### "Cannot find data/per_frame/"
→ Make sure you extracted to the correct location. The path should be:
```
chessboard-recon/data/per_frame/game2_per_frame/...
```
Not:
```
chessboard-recon/data/chessboard-data/per_frame/...
```

### "Running out of disk space"
→ You need at least 10 GB free space for:
- Raw data (~3-6 GB)
- Preprocessed data (~500 MB)
- Dataset splits (~500 MB)
- Virtual environment and packages (~1 GB)

### "Download is too slow"
→ Consider downloading only `per_frame/` (skip `PGN/` unless you need extra data)

## 📝 For Data Maintainers

If you need to update the shared data:

1. **Compress the data:**
   ```bash
   cd data
   zip -r chessboard-data.zip per_frame/ PGN/
   ```

2. **Upload to shared drive**

3. **Update the download link in this README**

4. **Update the dataset statistics if needed**

## 🔒 Data License & Citation

[ADD LICENSE INFORMATION IF APPLICABLE]
[ADD CITATION IF FROM EXTERNAL SOURCE]

---

**Need help?** Contact [YOUR EMAIL/SLACK/DISCORD]

