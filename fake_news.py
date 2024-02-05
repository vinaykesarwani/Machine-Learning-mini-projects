import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import classification_report
import re
import string

data_fake=pd.read_csv("Fake.csv")
data_true=pd.read_csv("True.csv")

data_fake["class"]=0
data_true["class"]=1

data_fake_mannual_testing=data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop([i],axis=0, inplace=True) 

data_true_mannual_testing=data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i],axis=0, inplace=True) 

data_fake_mannual_testing["class"]=0
data_true_mannual_testing["class"]=1

data_merge=pd.concat([data_fake, data_true], axis=0)

data=data_merge.drop(['title', 'subject', 'date'], axis=1)

data=data.sample(frac=1)# for random shuffling

data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

def wordopt(text):
    text=text.lower()
    text=re.sub('\[.*\]', "", text)
    text=re.sub('\W', " ", text)
    text=re.sub('https?://\S|www\.\S', "", text)
    text=re.sub('\n', '', text)
    text=re.sub('\w*\d\w*', '', text)
    return text

data["text"]=data["text"].apply(wordopt)
x=data['text']
y=data['class']

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.25, random_state=0)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization=TfidfVectorizer()
xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()

LR.fit(xv_train, y_train)

pred_lr=LR.predict(xv_test)

from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt=DT.predict(xv_test)

from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier(n_estimators=10)
RF.fit(xv_train, y_train)

pred_rf=RF.predict(xv_test)

def test(news):
    test_news={"text": [news]}
    new_x_test=pd.DataFrame(test_news)
    new_x_test["text"]=new_x_test["text"].apply(wordopt)
    new_x_test=new_x_test["text"]
    new_xv_test=vectorization.transform(new_x_test)
    print(LR.predict(new_xv_test))
    print(RF.predict(new_xv_test))
    print(DT.predict(new_xv_test))

test("21st Century Wire says As 21WIRE predicted in its new year s look ahead, we have a new  hostage  crisis underway.Today, Iranian military forces report that two small riverine U.S. Navy boats were seized in Iranian waters, and are currently being held on Iran s Farsi Island in the Persian Gulf. A total of 10 U.S. Navy personnel, nine men and one woman, have been detained by Iranian authorities. NAVY STRAYED: U.S. Navy patrol boat in the Persian Gulf (Image Source: USNI)According to the Pentagon, the initial narrative is as follows: The sailors were on a training mission around noon ET when their boat experienced mechanical difficulty and drifted into Iranian-claimed waters and were detained by the Iranian Coast Guard, officials added. The story has since been slightly revised by White House spokesman Josh Earnest to follow this narrative:The 2 boats were traveling en route from Kuwait to Bahrain, when they were stopped and detained by the Iranians.According to USNI, search and rescue teams from the Harry S. Truman Strike group were scrambled to aid the crew but stopped short after the crew was taken by the Iranians. We have been in contact with Iran and have received assurances that the crew and the vessels will be returned promptly,  Pentagon spokesman Peter Cook told AP. According to Persian Gulf TV: Farsi Island is one of the Iranian islands in the Persian Gulf, Bushehr province of Iran. There is an IRGC Navy base on this Island. The island has an area of about 0.25 km  and is completely restricted to public, due to top secret governmental activities. According to NBC News, U.S. State Department is in touch with Tehran officials and the Iranians recognize that the U.S. Navy straying off course was a mistake, and that the sailors will be released  within hours.  WAR INC: CNN s Wolf Blitzer wasted no time in ramping-up talk of military tension with Israeli-financed neocon Senator Tom Cotton.Neocon StuntAlready, the U.S. media, including CNN and FOX, are running with the talking point that,  this could not have come at a worse time for President Obama right before tonight s State Of Union speech, when he s trying to prove to the American people that Iran is a country that can be trusted to implement the historic nuclear deal. This latest Naval  controversy  also comes days before the implementation phase of the Iran Nuclear Deal. To say this is a coincidence might be naive.That said, could GOP and Israel-aligned members of the Pentagon or intelligence establishment have helped to engineer today s bizarre  mini-crisis  in order to help weaken U.S.-Iran relations, and by extension, Obama s controversial Iranian Nuclear Deal?This looks likely to be the case, as evidenced by the quick appearance of the Israel Lobby-sponsored, pro-war U.S. Senator Tom Cotton (R), placed by CNN to direct aggressive U.S. military talking points live on air as the story broke today. Cotton (photo, left) immediately called the event  hostile  and blamed Iran for the U.S. boat drifting into Iranian waters, and then blamed the  crisis  on President Obama, who he claims,  has emboldened the Iranian aggression. Cotton then goes on to tell a giant lie, on which his media handler, CNN s Wolf Blitzer, does not even blink, much less challenge Cotton s imaginary statement: The Iranians, who are largely responsible for killing our (American) soldiers in Iraq and Afghanistan. Cotton then went on to threaten Iran, saying that: These sailors and both vessels  need to be immediately released. If they are not released, then the Iran (nuclear) deal should not go forward, and military force will be on the table to retaliate for this act of aggression.  Cotton then proceeded to give a veiled (nuclear?) threat to Iran, saying that,  All (military) options should be on the table.  Would Washington s top operatives go so far as to engineer or exacerbate an international crisis such as this   by dispatching the boats off course, knowing full-well that Iran would not harm U.S. personnel, but using the incident to injure a lame duck? The answer is  yes , and they ve done it before In 1979, 52 American diplomats and citizens were held hostage for 444 days in what became known as the Iranian Hostage Crisis, which just happened to take place during a US presidential election cycle, with then president Jimmy Carter (D) running for reelection against challenger Ronald Reagan (R). The crisis, including a horrific failed rescue attempt, was used against Carter in the media. According to reports at the time Reagan campaign operatives had managed to do a backdoor deal with the elements of the new Iranian regime to  hold-off  releasing the hostages until after the election. In the end, Reagan won and took credit for that  victory .Not surprisingly, at the end of his prearranged CNN segment, Cotton invoked the  feckless foreign policy  of Jimmy Carter which  caused  the 1979 Iran Hostage Crisis, and compared that to President Obama s current policy.Of all the U.S. Officials CNN could have brought in on  short notice , they chose Senator Tom Cotton, the most hawkish and closest aligned to Israel. Clearly, this looks like a neocon stunt.Stay tuned for more updates.READ MORE IRAN NEWS AT: 21st Century Wire Iran Files")