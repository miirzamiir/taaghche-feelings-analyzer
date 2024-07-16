# Taaghche Feelings Analyzer


##### This project is Homework #3 of the NLP Course at the CE Department of Sharif University of Technology, lectured by [Dr. Ehsaneddin Asgari](https://github.com/ehsanasgari). The aim is to train some models on [taaghche](https://taaghche.com) comments. 

<br/>

### Data
For this project we used different datasets, including [taaghche sentiment analysis](https://www.kaggle.com/code/m3hrdadfi/taaghche-sentiment-analysis), [Arman](https://www.kaggle.com/datasets/shayanbemanian/arman-persian-ner-dataset), [Persian-NER](https://github.com/Text-Mining/Persian-NER), and we have crawled the website to collect some more data.
</h2>
 
### Requirements

- python
- Jupyter Notebook
- Additional libraries specified in `requirements.txt`
 
### Installation 

1. Clone the repository.<br/>
`git clone https://github.com/miirzamiir/taaghcheh-feelings-analyzer.git`

2. Install the dependencies.<br/>
`pip install -r requirements.txt`
`
### Models
For this project we have trained 4 models:
1. First one is a `Naïve Bayes` model which had been trained on [taaghche sentiment analysis](https://www.kaggle.com/code/m3hrdadfi/taaghche-sentiment-analysis). Here is the result of the trained model:<br/>Accuracy: 0.7341618882320847<br>
Precision (Macro): 0.7327678622348551<br>
Recall (Macro): 0.7341135534607575<br>
F1 Score (Macro): 0.732412323288311<br>
F1 Score (Micro): 0.7341618882320847<br>

2. For the second model, a transformer-based model named `pars-bert` has been trained. You can see the model [here](https://huggingface.co/hamedjahantigh/TaaghcheFeelingCommentAnalysis). Here is the result of the trained model:<br>F1 Score: 0.799<br>
Accuracy: 0.800<br>
Precision: 0.801<br>
Recall: 0.800<br>

3. For the third model we have trained a `HMM` model, which you can see its results:
 <br> Accuracy: 58.74% <br>
F1-macro score: 5.37% <br>
F1-micro score: 58.73% <br>
Precision-macro: 5.94% <br>
Precision-micro: 58.73% <br>
Recall-macro: 5.94% <br>
Recall-micro: 58.73% <br>

4. And finally, for the last model again we have used the `pars-bert` model for `BIO tags`. You can see the model [here](https://huggingface.co/hamedjahantigh/TaaghcheBIOTag). Here is the result of the model.<br>
F1 Score: 0.3534761203704847<br/>
Accuracy: 0.8980074083535573<br/>
Precision: 0.48302243550652585<br/>
Recall: 0.3239270904426163<br/> 

### Contributors
Here is the list of all contributors of this project:<br/>
[Amirmohamad Shakuri](https://github.com/miirzamiir)    -    [Hamed Jahantigh](https://github.com/HamedJahantigh-git)    -    [Zahra Maleki](https://github.com/Zzmaleki)
