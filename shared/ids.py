def detect(model, message):

    prediction = model.predict(message, verbose=0)

    score = float(prediction[0][0])

    if score > 0.5:
        return "ATTACK"
    else:
        return "BENIGN"