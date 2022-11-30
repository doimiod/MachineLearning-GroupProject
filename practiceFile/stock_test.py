import yfinance as yf

Address = 'practiceFile/'

start_date = '2011-12-03'
end_date = '2011-12-04'
ticker = 'TSLA'
data = yf.download(ticker, start_date, end_date)
data["Date"] = data.index

data = data[["Date", "Open", "High",
"Low", "Close", "Adj Close", "Volume"]]

data.reset_index(drop=True, inplace=True)
size = len(data)
print(size)
print(data)
data.to_csv("{}TSLA.csv".format(Address))

print("still continues")