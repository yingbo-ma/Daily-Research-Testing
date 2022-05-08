print("Gettting Model Layers Output...")

for layer_index in range(len(model.layers)): # model is the pre-defined NN model
    print("Layer Index: ", layer_index)
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
    print(intermediate_layer_model.predict(feature_vali))
    print("##################################################################")