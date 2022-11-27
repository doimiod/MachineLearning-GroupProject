import yfinance as yf
from matplotlib import pyplot as plt
from datetime import date, timedelta
import pandas as pd

#define the ticker symbol
tickerSymbol = 'TSLA'

#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

#get the historical prices for this ticker
# tickerDf = tickerData.history(period='1d', start='2010-1-1', end='2022-11-25')

Start = date.today() - timedelta(365*5)
Start.strftime('%Y-%m-%d')

End = date.today() - timedelta(2)
End.strftime('%Y-%m-%d')

Tesla = pd.DataFrame(yf.download(tickerSymbol, start=Start,end=End)['Adj Close'])  

#see your data
plt.plot(Tesla)
plt.show()