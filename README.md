# Hierarchical Attention Networks for Document Classification

This is a :fire: PyTorch :fire: implementation of the fantastic HAN Paper from CMU(***https://www.aclweb.org/anthology/N16-1174.pdf***) 
complete with Yelp Dataloaders and Minibatching Loading for Training 

## Architecture

![HAN Architecture](https://i.ibb.co/ygb4h6q/han-model-architecture.png)

The only modification is that I have swapped out GloVe for Elmo for improved performance :neckbeard: :rocket:

## Dataset

I have uploaded a sample dataset from the Yelp 2013 Reviews Dataset.
Download the complete dataset at [Yelp Reviews Dataset](https://www.yelp.com/dataset/download)




## Demo
:bowtie: I have also implemented a Streamlit App to interact with the model! :bowtie:

![Streamlit App](https://i.ibb.co/z4V303H/han-model-app.png)


## Instructions 

#### To Install Requirements

> `pip install -r requirements.txt`

#### To Train a New Model

> ` python train.py`

#### To Run the Demo Streamlit App
 
> ` streamlit run app.py`

## Improvements
:v: Pull requests are welcome for any improvements/features :v:

