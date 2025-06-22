from django.shortcuts import render,redirect
from django.http import HttpResponse
import pickle
from .models import Review
#import spacy
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LogisticRegression


# Create your views here.

def home(request):
    if request.method == 'GET':
        return render(request, 'home.html')
    if request.method == 'POST':
        # Get the review from the form
        review = request.POST.get('review')
        item= request.POST.get('item')
        rating = request.POST.get('rating')
        reviewobj = Review.objects.create(review_text=review,item=item,rating=rating)
        reviewobj.save()
        # Load the model and vectorizer from pickle files
        with open('model/logistic_regression_model.pkl', 'rb') as model_file:
            loaded_logistic_classifier = pickle.load(model_file)

        with open('model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            loaded_tfidf_vectorizer = pickle.load(vectorizer_file)

        # Vectorize the sample review using the loaded vectorizer
        review_tfidf = loaded_tfidf_vectorizer.transform([review])

        # Make a prediction using the loaded model
        prediction = loaded_logistic_classifier.predict(review_tfidf)

        # Interpret the prediction
        sentiment = "sorry! we will look into it." if prediction[0] == 0 else "glad you like it!."

        return render(request, 'home.html', {'review': review,'sentiment': sentiment})
    else:
        return render(request, 'home.html')



