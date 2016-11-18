def em_model(df_train, num_cluster):
    
    # return EM model
    
    df = pd.DataFrame(df_train,  index = df_train.index.tolist())
    gmm = mixture.GaussianMixture(n_components = num_cluster, covariance_type='full', random_state = 0)
    #em = mixture.GMM(n_components = num_cluster, random_state = 0)
    #model = em.fit(df_train)
    model = gmm.fit(df_train)
    return(model)

