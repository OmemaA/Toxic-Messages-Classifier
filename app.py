
import streamlit as st
import json
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

@st.cache
def Import(): 
    import classifier
    model = classifier.buildModel()
    return classifier, model

classifier, model = Import()

html_temp = """ 
        <h1 style='text-align: center; color: Black;'>Toxic Messages Classifier</h1>
        <div style = "background-color:tomato; padding:10px">
        <h3 style="text-align: center;color:white;"> Please note that your messages are being filtered and the use of inappropriate language will be not tolerated.<br> 
        Repeat offenders will be banned</h3>
        </div>
        """
st.markdown(html_temp, unsafe_allow_html=True)

data = json.loads(open(r'data_tolokers.json','r').read()) #change path accordingly

train = []
for row in data:
    train.append(row['dialog'][0]['text'])

bot = ChatBot(name = 'PyBot', read_only = False)
# corpus_trainer = ChatterBotCorpusTrainer(bot) 
# corpus_trainer.train('chatterbot.corpos.english') 

# corpus_trainer = ListTrainer(bot)
# corpus_trainer.train(train)
def get_text():
    input_text = st.text_input("You: ")
    return input_text 


user_input = get_text()

if user_input:
    values = classifier.check(model, user_input)
    category = [classifier.labels[i] for i, value in enumerate(values) if value == 1]
    if len(category) != 0:
        classifier.offence += 1
        if classifier.offence >= 3:
            for i,c in enumerate(category):
                st.write(i+1,c)
            html = """ 
                <div>
                <h2 style="text-align: center;color:red;"> 
                You have been banned!
                </h2>
                </div>
                """
            st.markdown(html, unsafe_allow_html=True)
            st.stop()
            
        else:
            st.write("""
            Your message was undelivered due to inappropriate content. The following categories were detected:
            """)
            for i,c in enumerate(category):
                st.write(i+1,c)
            html = """ 
                <div>
                <h2 style="text-align: center;color:red;"> 
                Warning: Repeat offenders will be banned
                </h2>
                </div>
                """
            st.markdown(html, unsafe_allow_html=True)
    else:
        st.text_area("Bot:", value=bot.get_response(user_input), height=200, max_chars=None, key=None)
else:
    st.text_area("Bot:", value="Please type a message to start talking to the bot", height=200, max_chars=None, key=None)

