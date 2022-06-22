# MultiSpeech

This is a PyTorch implementation of [MultiSpeech: Multi-Speaker Text to Speech with Transformer](https://arxiv.org/pdf/2006.04664.pdf)

![model](https://user-images.githubusercontent.com/61272193/175074608-12b98fbd-c102-4c55-af08-d2676787650f.png)

# Train on your data
In order to train the model on your data, follow the steps below 
### 1. data preprocessing 
* prepare your data and make sure the data is formatted in an PSV format as below without the header
```
speaker_id,audio_path,text,duration
0|file/to/file.wav|the text in that file|3.2 
```
The speaker id should be integer and starts from 0
* make sure the audios are MONO if not make the proper conversion to meet this condition

### 2. Setup development environment
* create enviroment 
```bash
python -m venv env
```
* activate the enviroment
```bash
source env/bin/activate
```
* install the required dependencies
```bash
pip install -r requirements.txt
```

### 3. Training 
* update the config file if needed
* train the model 
  ```bash
  python train.py --train_path train_data.txt --test_path test_data.txt --checkpoint_dir outdir --epoch 100 --batch_size 64
  ```
