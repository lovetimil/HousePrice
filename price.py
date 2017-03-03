


import pandas as pd
import numpy as np
from PriceModel import PriceModel


# train
pricepredict = PriceModel()

pricepredict.train(file)
pricepredict.test()
pricepredict.save("./pricemodel/")



#predict


precepredict_pp = PriceModel().load("./pricemodel/")

data = ["58ac57672dedf740ebcb05ea","2015年09月（築2年）","15枚","空家","平成23年築。 駐車場2台可。 南西向きで陽当たり?通風良好。","９４.５７坪",
       "４ＬＤＫ","南","269.89m2","叡山電鉄鞍馬線 京都精華大前駅 徒歩8分京都府京都市左京区岩倉幡枝町1135,叡山電鉄鞍馬線","京都精華大前駅 徒歩,京都府京都市左京区岩倉幡枝町1135"]

precepredict_pp.predict(data)
