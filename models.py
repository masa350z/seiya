from transformers import TFBertModel
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.models import Model


class Attention(Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.d1 = Dense(units, activation='relu')
        self.d2 = Dense(units, activation='relu')
        self.d3 = Dense(units, activation='sigmoid')

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.d3(x)
        
        return inputs*x


class Simple_Boat_Bert(Model):
    def __init__(self, num_classes, bert_model='bert-base-uncased'):
        super(Simple_Boat_Bert, self).__init__(name='boat_bert')

        self.bert_model = TFBertModel.from_pretrained(bert_model)
        self.output_layer = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.bert_model(inputs)[1]
        x = self.output_layer(x)
        
        return x
