from flask import Flask, render_template, request
from score import classify
import plotly.express as px
import pandas as pd
import re
import plotly
import pickle
import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
VOCAB = pickle.load(open('data/vocab.pkl', 'rb'))
MODEL = torch.load('weights/best_model.pt', map_location=device)
MODEL.to(device)

app = Flask(__name__)


def get_plot(probs_pd):
    fig = px.bar(probs_pd, y='Percentage', height=700)
    plot_div = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
    return plot_div


@app.route('/', methods=['GET', 'POST'])
def index():
    news = ''
    if request.method == 'POST':
        news = request.form.get('get_news')
    if (news != '') and (re.sub(r'\s', '', string=news) != ''):
        probs_pd = pd.DataFrame.from_dict(classify(news.strip(), MODEL, VOCAB, device),
                                          orient='index',
                                          columns=['Percentage'])
        plot_div = get_plot(probs_pd)
        info = {'news': news, 'category': probs_pd.Percentage.idxmax(), 'percent': probs_pd.Percentage.max()}
    else:
        plot_div = None
        info = None
    return render_template('index.html', info=info, plot=plot_div)


if __name__ == "__main__":
    app.run(debug=True)
