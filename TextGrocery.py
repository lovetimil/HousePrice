# coding:utf-8
from collections import defaultdict
import cPickle
import os
import re
from bs4 import BeautifulSoup
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import jieba
#from base import *
from sklearn.feature_extraction.text import TfidfVectorizer
#from stop_words import list_stop_words

__all__ = ['GroceryTextConverter']

Feat_N = 4000


def _dict2list(d):
    if len(d) == 0:
        return []
    m = max(v for k, v in d.iteritems())
    ret = [''] * (m + 1)
    for k, v in d.iteritems():
        ret[v] = k
    return ret


def _list2dict(l):
    return dict((v, k) for k, v in enumerate(l))


class GroceryTextPreProcessor(object):
    def __init__(self):
        # index must start from 1
        self.tok2idx = {'>>dummy<<': 0}
        self.idx2tok = None

    @staticmethod
    def _default_tokenize(text):
        # return jieba.cut(text.strip(), cut_all=True)
        return jieba.cut(text.strip(), cut_all=False)

    def purification(self, text, remove_stopwords=False):
        #text = BeautifulSoup(text).get_text()
        #text = re.sub("[\d\s+\.\!\/_,$%^*()-+\"\'╱\\\?\,@#$]+|[《》●+——！，。？、~@#￥%……&*（）「」:]+".decode("utf8"),
        #              "".decode("utf8"), text)
        try:
            text = re.sub(u"[\d\s+\.\!\/_,$%^*()-+\"\'╱\\\?\,@#$]+|[《》●+——！，。？、~@#￥%……&*（）「」:]+",
                          u"", text.decode("utf-8").strip())

            # text = re.sub("[a-zA-Z]","",text)
            if remove_stopwords:
                pass
            return text
        except:
            return u""


    def preprocess(self, text, custom_tokenize=None):
        text = self.purification(text)
        if custom_tokenize is not None:
            tokens = custom_tokenize(text)
        else:
            tokens = self._default_tokenize(text)

        ret = ' '.join(tokens)
        return ret
    def preprocess_list(self,textlist,cutom_tokenize=None):
        return map(self.preprocess,textlist)
    def save(self, dest_file):
        self.idx2tok = _dict2list(self.tok2idx)
        config = {'idx2tok': self.idx2tok}
        cPickle.dump(config, open(dest_file, 'wb'), -1)


    def load(self, src_file):
        config = cPickle.load(open(src_file, 'rb'))
        self.idx2tok = config['idx2tok']
        self.tok2idx = _list2dict(self.idx2tok)
        return self


class GroceryFeatureGenerator(object):
    # def __init__(self,name):
    def __init__(self,maxfeatures = 200):
        # self.name = name
        self.stop_words = set([])
        self.max_words = maxfeatures
        self.tfidf = TfidfVectorizer(max_features=self.max_words,
                                     ngram_range=(1, 3), sublinear_tf=True)

    def settfidf(self, stopwords_path=None, max_features=4000):
        if stopwords_path != None:
            self.get_stopwords(stopwords_path)
            self.tfidf.set_params(stop_words=list(self.stop_words))
        if max_features != 4000:
            self.tfidf.set_params(max_features=max_features)
        return self

    def get_stopwords(self, stop_words_path):
        try:
            with open(stop_words_path, 'r') as fin:
                contents = fin.read().decode('utf-8')
        except IOError:
            raise ValueError("the given stop words path %s is in invalid." % (stop_words_path))
        for line in contents.splitlines():
            self.stop_words.add(line.strip())
        print("\nsuccess in getting stopwords\n")

    def fit_transform(self, textlist):
        # self.settfidf('resource/stop_words.txt')
        # self.settfidf(os.path.join(path,'stop_words.txt'))
        # if isinstance(textlist,list):
        if 1:
            tf = self.tfidf.fit_transform(textlist)
        # with open(self.tfidfpath,'w') as fout:
        #    pickle.dump(tf)
        return tf.toarray()

    #    def load_tfidf(self):
    #        try:
    #            with open(self.tfidfpath,'r') as fin:
    #                self.tfidf = pickle.load(fin)
    #        except IOError:
    #            raise ValueError("the %s path is invalid."%(self.tfidfpath))

    def transform(self, textlist):
        # self.load_tfidf()
        if isinstance(textlist, list):
            return self.tfidf.transform(textlist)

    def save(self, dest_file):
        config = self.tfidf
        cPickle.dump(config, open(dest_file, 'wb'), -1)

    def load(self, src_file):
        self.tfidf = cPickle.load(open(src_file, 'rb'))
        return self


class Textmodel(object):
    def __init__(self, words_num=100):
        self.text_preprocessor = GroceryTextPreProcessor()
        self.tfidf_processor = GroceryFeatureGenerator()
        self.words_num = words_num

    def train(self, X):
        tokenize_word = self.text_preprocessor.preprocess_list(X)
        ret = self.tfidf_processor.fit_transform(tokenize_word)
        return self
    def fit_transform(self, X):
        tokenize_word = self.text_preprocessor.preprocess_list(X)
        ret = self.tfidf_processor.fit_transform(tokenize_word)
        return ret

    def transform(self, X):
        tokenize_word = self.text_preprocessor.preprocess_list(X)
        ret = self.tfidf_processor.transform(tokenize_word)
        return ret
    def save(self,path):
        self.text_preprocessor.save("%s_%s"%(path,"pre"))
        self.tfidf_processor.save("%s_%s"%(path,"tf"))
        return self
    def load(self,path):
        self.text_preprocessor = self.text_preprocessor.load("%s_%s"%(path,"pre"))
        self.tfidf_processor = self.tfidf_processor.load("%s_%s"%(path,"tf"))
        return self

class TextBatch(object):
    def __init__(self,feature_num=1):
        self.pre = 'textfeature'
        self.textprocessorlist = []
        self.feature_num =4
    def train(self,ndarray):
        (a,b) = ndarray
        for i in range(b):
            textprocessor = Textmodel()
            textprocessor.train(ndarray[:,i])
            self.textprocessorlist.append(textprocessor)
        return self
    def fit_transform(self,ndarray):
        (a,b) = ndarray.shape
        ret_feature = None
        for i in range(b):
            textprocessor = Textmodel()
            ret = textprocessor.fit_transform(ndarray[:,i])
            #print("ret",ret.shape)
            self.textprocessorlist.append(textprocessor)
            if ret_feature is None:
                ret_feature = ret
            else: ret_feature = np.hstack((ret_feature,ret))
        return ret_feature
    def transform(self, ndarray):
        (a,b) = ndarray.shape
        ret_feature = None
        for i in range(b):
            ret = self.textprocessorlist[i].transform(ndarray[:, i])
            #print("ret",ret.shape)
            (a,b) = ret.shape
            ret = ret.toarray().reshape(a,b)
            if ret_feature is None:
                ret_feature = ret
            else: ret_feature = np.hstack((ret_feature,ret))
        return ret_feature
    def save(self,path):
        l = len(self.textprocessorlist)
        for i in range(l):
            path_ = "%s_%s"%(path,i)
            self.textprocessorlist[i].save(path_)
        return self
    def load(self,path):
        for i in range(self.feature_num):
            path_ = "%s_%s"%(path,i)
            self.textprocessorlist.append( Textmodel().load(path_))

        return self

class StringT(object):

    def __init__(self):
            self.dic = defaultdict(int)
            self.pre = 0
    def String2id(self,s):
        id = s
        if(isinstance(s,str)):
            id = self.dic.get(s,-1)
            if(id == -1) :
                id = self.pre+1
                self.pre += 1
                self.dic[s]=  id
        return id
    def getPortaitNum(self,ss):
        try:
            s = re.search(r'\d+',ss)
            if s:
                return int(s.group(0))
        except:
            return None
        return None

    def getArea(self,ss):
        try:
            s = re.search(r'\d+',ss)
            if s:
                return int(s.group(0))
        except:
            return None
        return None

    def getPrice(self,ss):
        try:
            ss = re.sub(r'"|,',"",ss)
            return int(ss)
        except:
            return None
        return None

    def getYears(self,ss):
        "1999年04月（築18年）"
        from datetime import datetime
        year = datetime.now().date().year
        now = datetime.now().date().year
        interval = 0
        try:
            s = re.search(r'\d+',ss)
            if s:
                year = int(s.group(0))
            interval = now - year
        except:
            interval = 0
        return interval


if __name__ == "__main__":

    textmodeltool = Textmodel(words_num = 50)

    text =['''<div>
<div><img src="http://cdnpic.highlights.tw/group1/M00/81/01/CgEBBFdywUuABF1jAABQwRYS9mM548.jpg" /></div>

<p><strong>&gt;&gt;&gt;</strong><strong>LPL2016夏季賽報道專題</strong></p>

<p>LPL夏季賽的戰鼓即將於5月26日17時正式打響,12支中國頂級《英雄聯盟》戰隊將為LPL夏季賽冠軍獎盃,以及2016《英雄聯盟》全球總決賽參賽名額發起衝擊。目前,常規賽階段的96場BO3的所有賽程正式確定,比賽周期為5月26日-8月7日。2016LPL夏季賽常規賽將分為三個比賽階段進行。</p>

<p>下面是LPL夏季賽 Snake vs OMG比賽情況</p>

<div><img src="http://cdnpic.highlights.tw/group1/M00/81/01/CgEBBFdywUyATWcZAACeL3y0xuo736.jpg" /></div>

<p>來到雙方的第二局比賽,本局VG拿到了對線強勢的千珏搭配巨魔的陣容,而EDG拿出的陣容則比較偏向於前中期的遊走為隊友早點打開局勢。比賽開始EDG選擇換線想要保證DFET前期的發育,前期中路的一波集結讓冰女丟掉閃現,在一波卡牌6級大招飛配合隊友直接拿到了錘石的一血。</p>

<p>&nbsp;</p>

<div><img src="http://cdnpic.highlights.tw/group1/M00/81/01/CgEBBFdywU2ALtGYAAClRkCOkG4045.jpg" /></div>

<p>前中期,EDG的中野都非常針對VG的雙人路,下路的優勢讓卡牌得以去邊路帶線推塔VG的節奏逐漸陷入被動,經濟也開始慢慢落後。急於打開局面的VG利用千珏和冰女強殺卡牌,被趕來的巴德救下,隨即EDG一波反打燼拿下三殺打出一波0換5,EDG獲得大優勢開始全線起飛。</p>

<p>&nbsp;</p>

<div><img src="http://cdnpic.highlights.tw/group1/M00/81/01/CgEBBFdywU2AEkLOAACS6UP5mzs797.jpg" /></div>

<p>憑藉全隊優勢的EDG在小龍處發起團戰再次打出一波1換3,VG雖然拿下了一條火龍,但是面對敵方裝備的碾壓勝算顯得異常渺茫。EDG的運營讓自己的節奏開始加快想要繼續擴大優勢,但一波團戰C位被秒,讓VG有了喘息的機會,隨後雙方互換大龍和遠古龍,一波團戰後EDG直衝高地推掉水晶獲得了BO3的勝利。</p>

<p>【更多電競賽事新聞及遊戲動態,微信關注太平洋遊戲網(pcgames_com_cn),每天推薦重磅遊戲新聞哦~】</p>
</div>
''','''<div>有星巴克店員爆料,很多台灣人都不知道星巴克的「正確點餐方式」!<br />
<br />
今天小編為你總結星巴克點餐全攻略,教你玩轉星巴克的餐牌!<br />
<br />
進店之前你要知道的<br />
<br />
❶ 杯型<br />
<br />
很多朋友點餐的時候都習慣叫小、中、大杯,其實在國內,常規的依次是Tall(中杯),Grande(大杯),Venti(超大杯)。<br />
<br />
搞不清楚杯型,就很容易想點中杯變小杯,想點超大杯變中杯啦~<br />
<br />
點咖啡的情況下,中杯一般默認加一份濃縮,大杯和超大杯都是兩份濃縮,所以超大杯只是多了一點水或奶,感覺還虧了呢!<br />
<br />
<img src="http://cdnpic.highlights.tw/group1/M00/25/B5/CgEBBFdQ35yAVk80AACmGVA_an4938.jpg" />
<div><br />
❷ 正確的點餐方式<br />
<br />
1、雖然現在一般會默認用紙杯,但你可以選擇馬克杯,或者自帶杯哦(保溫杯什麼的都可以,重點是自帶杯可以省10元)!<br />
<br />
2、含咖啡因的飲品可以選擇低因咖啡豆(decaf),還可以買無咖啡因的飲品。<br />
<br />
3、咖啡的濃度也可以選擇,想濃一點的可以讓服務員多加一份濃縮(每加一份濃縮要加20元呢!)。<br />
<br />
4、糖漿的口味也可以選擇。舉個栗子,你可以把香草拿鐵裡的香草糖漿換成其他味道的,榛子,杏仁,焦糖(但是要收費的~)。<br />
<br />
5、做咖啡用的奶製品,除了默認的牛奶,還可以換成豆奶,脫脂牛奶,半全脂半脫脂呢~<br />
<br />
6、你甚至可以要求拿鐵少奶泡,摩卡多奶油,卡布多牛奶,連牛奶的溫度,你都可以有要求~<br />
<br />
7、以上的要求,在點單的時候一次性告訴夥伴,就省去了點單時多餘的對話,還可以裝逼:&ldquo;我想要一杯這裡喝的,低咖啡​​因的,單份濃度的,杏仁,脫脂牛奶做的,特別熱的,中杯拿鐵。&rdquo;<br />
<br />
所有要求,店員都能準確地在杯子上記下來~<br />
<br />
<img src="http://cdnpic.highlights.tw/group1/M00/25/B5/CgEBBFdQ356AF_HJAACXU2GxVio290.jpg" /><br />
❶ 咖啡類<br />
<br />
濃縮咖啡:做其他咖啡的基礎,分量很小,分單份23ml;雙份45ml。<br />
<br />
美式咖啡:濃縮咖啡+水<br />
<br />
本週咖啡:星巴克每週會換一種咖啡豆,本週咖啡其實是用當週的咖啡豆做的蒸餾咖啡,比美式咖啡更純正一些。可免費加牛奶,牛奶分量由自己決定,但加熱牛奶要加錢呢!<br />
<br />
卡布奇諾:濃縮咖啡+牛奶+奶泡。比例:1:1:1<br />
<br />
拿鐵:濃縮咖啡+牛奶+奶泡。比例:1:2:1<br />
<br />
焦糖瑪奇朵:濃縮咖啡+牛奶+奶泡。最後淋上焦糖醬。焦糖瑪奇朵的奶泡分量比拿鐵要多。<br />
<br />
摩卡:濃縮咖啡+牛奶+摩卡醬。最後加上鮮奶油。<br />
<br />
❷ 茶類<br />
<br />
冰搖紅莓黑加侖茶<br />
<br />
檸檬茶類:有點偏甜。<br />
<br />
紅茶拿鐵、抹茶拿鐵:這兩款飲品中並沒有咖啡因,很多人以為拿鐵就是咖啡,其實拿鐵(latte)在意大利語裡是&ldquo;鮮奶&rdquo;的意思,所以咖啡拿鐵就是牛奶加咖啡;紅茶拿鐵、抹茶拿鐵就是在牛奶中加入紅茶或抹茶。<br />
<br />
星巴克的這兩款飲品中會加入糖漿噢,所以可以根據自己的喜好要求夥伴控制甜度,和茶的濃度。<br />
<br />
(推荐一個隱形菜單:紅茶拿鐵其實就是奶茶,喜歡喝伯爵紅茶的朋友不妨把紅茶換伯爵,試一下伯爵拿鐵噢!)<br />
<br />
<img src="http://cdnpic.highlights.tw/group1/M00/25/B5/CgEBBFdQ35-AeJjoAABcAzY3dz4244.jpg" />
<div><br />
❸ 星冰樂<br />
<br />
星冰樂的共同特點就是甜,因為都需要加星巴克特有的星冰樂糖漿來混合。一般需要兩種或兩種以上的糖漿,還需要加奶油。不喜歡甜的可以半糖。<br />
<br />
抹茶星冰樂:長得好看,所以很多不喜歡喝咖啡又愛拍照的朋友都愛翻它牌。<br />
<br />
<img src="http://cdnpic.highlights.tw/group1/M00/25/B5/CgEBBFdQ35-AKkdnAADrCWs65dI465.jpg" /><br />
香草風味星冰樂、芒果西番蓮果茶星冰樂:香草風味的星冰樂,加上攪打奶油;芒果西番蓮果汁加細冰攪打而成的星冰樂。<br />
<br />
摩卡可可碎片星冰樂:這款星冰樂是含有咖啡因的,可可碎片還會加上顆粒的可可,在奶油上會加巧克力醬,相對最甜。<br />
<br />
焦糖濃縮咖啡星冰樂:這款也是含有咖啡因的星冰樂,在咖啡星冰樂上淋上糖漿。<br />
<br />
(講真,小編覺得星冰樂就只有一個字&ldquo;甜&rdquo;)<br />
<br />
<br />
隱藏菜單<br />
<br />
其實隱藏菜單就是通過變化甜度、增加或變換焦糖、奶油等來實現的。<br />
<br />
<strong>改變甜度:</strong>無糖或半塘<br />
<br />
<strong>風味糖漿:</strong>香草、榛果等,一杯普通的拿鐵加份榛果糖漿就變榛果拿鐵了,加焦糖就變焦糖拿鐵了~(不過要加錢的...)<br />
<br />
<strong>增加咖啡:</strong>加一份濃縮咖啡<br />
<br />
<strong>增減鮮奶油:</strong>喜歡奶油的朋友,點單時記得說要多奶油噢!聽說還可以點一杯奶油的~<br />
<br />
<strong>換茶包:</strong>紅茶拿鐵可以換個伯爵紅茶茶包就成了伯爵拿鐵了,還可以換鐵觀音試試呢!<br />
<br />
<strong>選擇奶類:</strong>全脂牛奶、脫脂奶、豆奶<br />
<br />
<strong>星巴克點餐全攻略到此為止!下次不妨試一下自己搭配一杯飲品啊!也許會有出乎意料之外的驚喜也說不定!喜歡這篇文章的話,就點個讚分享出去吧!</strong>
<div>文章來源</div>
</div>
</div>
</div>''']

    textmodeltool.train(text)
    text_test = [u"Notepad++中文版是一款优秀的支持语法高亮的开源免费的纯文本编辑器，在文字编辑方面与Windows写字板功能相当，Notepad++更是程序员们编写代码的利器。"]
    ret = textmodeltool.fit(text_test)
    print ret