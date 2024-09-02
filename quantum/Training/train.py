

def train(X, Y, model, optimiser, iteration, lossfn, callback = None):
    for i in range(iteration):
        optimiser.zero_grad()

        prediction = model(X)

        loss = lossfn(Y, prediction)

        loss.backward()

        optimiser.step()

        if callback != None: 
            callback(model, loss)