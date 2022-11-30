# Algomo

## Multilingual Sentence Grouping

The target of this project is to build a multilingual (100+ languages) sentence grouping model by using transfer learning with TensorFlow Hub.

The model will match question-answer pairs across languages, by assigning a "faq_id" on them. The data used is a CSV file with question-answer pairs in different languages. The data is built by another project https://github.com/hsuning/qna-web-scraping.

**Solution Overview**

- Sentences level data
- Embedding sentences with Multilingual Universal Sentence Encoder on Tensorflow Hub (Transfer Learning)
- Computing sentence level semantic similarity with embeddings vectors
- Grouping the sentences using algorithm built by myself

### Folder Structure
    .
    ├── data                                       # Data used by notebook
    ├── installation                               # Files for development environment installation on Apple Chip
    ├── Multilingual_Sentences_Grouping.ipynb      # Codes with solution explainations
    ├── Multilingual_Sentences_Grouping.html       # HTML version of codes with solution explainations
    ├── LICENSE
    └── README.md

### Built With
This section list all frameworks/libraries used.
- joblib==1.1.0
- numpy==1.23.3
- pandas==1.4.4
- scikit_learn==1.1.3
- tensorflow==2.10.0
- tensorflow-hub==0.12.0
- tensorflow-text==2.10.0
- universal-sentence-encoder-cmlm/multilingual-base: <https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1>

<!-- GETTING STARTED -->
## Getting Started

### Run the Indexing-API with Docker Playground on the Cloud
You can first test the indexing API that wrap the model without installing any things. The only thing you need is a web browser. In **Indexing-API-test-on-cloud.pdf**, you can find the same instruction with **screenshots**.

1. Open a web browser and go to <https://labs.play-with-docker.com/>

2. Login with your docker account and click on the start button

4. Add a new instance

5. Input the command below to pull the image from my Docker Hub (around 1 minute):
```
docker pull hsuningchang/indexing-linux:latest
```
> The link to my Docker Hub is <https://hub.docker.com/r/hsuningchang/indexing-linux.> I will delete it after your testing.

6. Once the image is downloaded, input the command below and wait for server to start (around 1.5 minute as it will download modules from Tensorflow Hub)
```
docker run -p 5555:5555 hsuningchang/indexing-linux:latest
```

7. The following message indicates that the server starts up:
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5555 (Press CTRL+C to quit)
```

8. Click on the “OPEN PORT” button on the top and input “5555”. Don’t forget to allow the pop-ups in your browser.
    
9. A page with “Hollo friends!” message shows. The URL of the page would be something similar to <http://ip172-18-0-4-ce37ldm3tccg009angdg-5555.direct.labs.play-with-docker.com/>
    
10. Then add another new instance
    
11. Input the command below for testing. Don’t forget to modify the URL part (red colour). The response time is generally lower than 3 seconds.
```
curl --request POST \
--header 'Content-Type: application/json' \
--data '{"query":"credit card", "top_n":20}' \
--url http://ip172-18-0-9-ce38kgm0qau0009v79o0-5555.direct.labs.play-with-docker.com/predict
```

> Please feel free to modify the –data part to test with other query or get more results.


12. The output result will be :

```
{"0":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-de"},"1":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-es"},"2":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-eu"},"3":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-at"},"4":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-fr"},"5":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-it"},"6":{"question":"Wie kann ich mein Konto schließen?","Ranking":2.0,"FAQ_id":40,"locale":"de","market":"de-at"},"7":{"question":"Wie kann ich mein Konto schließen?","Ranking":2.0,"FAQ_id":40,"locale":"de","market":"de-de"},"8":{"question":"How to protect my account?","Ranking":3.0,"FAQ_id":90,"locale":"en","market":"en-it"},"9":{"question":"How to protect my account?","Ranking":3.0,"FAQ_id":90,"locale":"en","market":"en-de"}}
```

### Installation
The whole solution is built on MacBook M1 or M2. If you are using the same computer, please follow the instruction below for installation.
If you are using other computer, you might need to modify the notebook and the package importion part in notebook.

In this section, we will create a new Conda environment and install all the necessary packages. At the end of the installation, you can follow the instruction to test the API.

1. Open a terminal and go to the folder **/installation**:
```sh
cd Hsuning_Chang_ml_eng/installation
```

2. Create an environment with ```environment.yaml``` file and activate it:

> Please note that you can change the environment name in this file.

```sh
  conda env create --file=environment.yaml
  conda activate multilingual-indexing
```

3. Update the pip and install all the packages with ```requirements.txt```:

```sh
  pip install --no-cache-dir -U pip
  pip install -r requirements.txt
```


### Sentence Encoder and Closest Matches
Modify the file paths in the first cell and run the whole notebooks.

```
input_file_path = 'data/extracted_n26_new.csv'
output_file_path = 'data/closest_matches.csv'
output_model_path = 'closest_match_model.pkl'
```

<!-- LICENSE -->

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.
