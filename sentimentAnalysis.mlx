//DATA EXTRACTION
Filename =TwitterComments.txt
Str=extractfileText(‘TwitterComments.txt’)

//PREPROCESSING OF THE DATA :
documents = tokenizedDocument(str)
documents = erasePunctuation(documents)
documents = removeStopWords(documents)
documents = lower(documents)
//LOAD PRETRAINED WORD EMBEDDING :
emb = fastTextWordEmbedding
//LOAD OPINION LEXICON :
data = readLexicon
function data = readLexicon

% Read positive words
fidPositive = fopen(fullfile('opinion-lexicon-English','positive-words.txt'))
C = textscan(fidPositive,'%s','CommentStyle',';')
wordsPositive = string(C{1})

% Read negative words
fidNegative = fopen(fullfile('opinion-lexicon-English','negative-words.txt'))
C = textscan(fidNegative,'%s','CommentStyle',';')
wordsNegative = string(C{1})
fclose all

% Create table of labeled words
words = [wordsPositive;wordsNegative]
labels = categorical(nan(numel(words),1))
labels(1:numel(wordsPositive)) = "Positive"
labels(numel(wordsPositive)+1:end) = "Negative"

data = table(words,labels,'VariableNames',{'Word','Label'})

end
idx = data.Label == "Positive"
head(data(idx,:))
idx = data.Label == "Negative"
head(data(idx,:))
//PREPARE DATA FOR TRAINING 
idx = ~isVocabularyWord(emb,data.Word)
data(idx,:) = []
numWords = size(data,1)
cvp = cvpartition(numWords,'HoldOut',0.1)
dataTrain = data(training(cvp),:)
dataTest = data(test(cvp),:)
wordsTrain = dataTrain.Word
XTrain = word2vec(emb,wordsTrain)
YTrain = dataTrain.Label
//TRAIN A SENTIMENT CLASSIFIER
mdl = fitcsvm(XTrain,YTrain)
//TEST CLASSIFIER
wordsTest = dataTest.Word
XTest = word2vec(emb,wordsTest)
YTest = dataTest.Label
//PREDICT THE SENTIMENT LABELS OF THE WORD VECTORS
[YPred,scores] = predict(mdl,XTest)

//VISUALIZE THE CLASSIFICATION ACCURACY IN A CONFUSION BOX
figure
confusionchart(YTest,YPred)
//VISUALIZE THE CLASSIFICATION IN THE WORD CLOUDS
figure
subplot(1,2,1)
idx = YPred == "Positive"
wordcloud(wordsTest(idx),scores(idx,1))
title("Predicted Positive Sentiment")

subplot(1,2,2)
wordcloud(wordsTest(~idx),scores(~idx,2))
title("Predicted Negative Sentiment")
//REMOVE THE WORDS FROM THE DOCUMENTS THAT DO NOT APPEAR IN THE WORD EMBEDDING EMB 
idx = ~isVocabularyWord(emb,documents.Vocabulary)
documents = removeWords(documents,idx)

words = documents.Vocabulary
words(ismember(words,wordsTrain)) = []

vec = word2vec(emb,words)
[YPred,scores] = predict(mdl,vec)

figure
subplot(1,2,1)
idx = YPred == "Positive"
wordcloud(words(idx),scores(idx,1))
title("Predicted Positive Sentiment")

subplot(1,2,2)
wordcloud(words(~idx),scores(~idx,2))
title("Predicted Negative Sentiment")
//CALCULATING THE SENTIMENT SCORES
results=0
compoundScores = vaderSentimentScores(documents)
compoundScores(1:10)
results=sum(compoundScores)
if (results<0)
disp( ‘ Disbalance in Sentiments detected ‘)
end
