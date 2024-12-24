#Any line filled with "______" indicates the presence of a separate cell

!pip install python-dotenv
__________________________
from openai import OpenAI
from dotenv import dotenv_values
import matplotlib.pyplot as plt
_______________________________
from google.colab import drive
drive.mount('/content/drive')
import glob
from sklearn.manifold import TSNE
import numpy as np
____________
%cd /content/drive/MyDrive/Colab\ Notebooks
____________________________________________
api_key = dotenv_values(".env")["OpenAI_APIkey"]
client = OpenAI(api_key = api_key)
response = client.embeddings.create(input = ["prince","princess","sofa","throne","man","woman"], model = "text-embedding-3-small")

X = np.array([response.data[0].embedding,response.data[1].embedding,response.data[2].embedding,response.data[3].embedding,response.data[4].embedding,response.data[5].embedding])
X_embedded = TSNE(n_components=2,perplexity=1.2).fit_transform(X)
X_embedded

prince = X_embedded[0]
princess = X_embedded[1]
sofa = X_embedded[2]
throne = X_embedded[3]
man = X_embedded[4]
woman = X_embedded[5]

plt.scatter(prince[0],prince[1],color="Crimson",label = "Prince")
plt.scatter(princess[0],princess[1],color="DarkRed",label = "Princess")
plt.scatter(sofa[0],sofa[1],color = "DodgerBlue",label = "Sofa")
plt.scatter(throne[0],throne[1],color = "MediumBlue",label = "Throne")
plt.scatter(man[0],man[1],color = "SpringGreen",label = "Man")
plt.scatter(woman[0],woman[1],color = "Green",label = "Woman")
plt.legend()

