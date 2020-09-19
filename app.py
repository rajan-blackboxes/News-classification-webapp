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


def get_models(device_name, model_path='weights/best_model.pt', vocab_path='data/vocab.pkl'):
    vocab = pickle.load(open(vocab_path, 'rb'))
    model = torch.load(model_path, map_location=device_name)
    model.to(device)
    return vocab, model


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
        vocab, model = get_models(device)
        probs_pd = pd.DataFrame.from_dict(classify(news.strip(), model, vocab, device),
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
