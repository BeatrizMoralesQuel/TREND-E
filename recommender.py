def user_input():

    ######## Import libraries ########
    import pandas as pd
    import numpy as np
    from random import randint
    import pickle
    import streamlit as st
    ######## Import data ######## 

    asos = pd.read_csv('data/asos_clusters.csv')
    
    
    ######## Valid input options ########
   
    valid_producttype = list(asos['#search'].unique())
    valid_brand = list(asos['#brand'].unique())
    valid_colour = list(asos['#colour'].unique())

    
    ######## User input ######## 
    producttype = st.selectbox('Product Type: ', valid_producttype, key='1')
    brand  = st.selectbox('Brand: ', valid_brand, key='2')
    colour = st.selectbox('Colour: ', valid_colour, key='3')
    price  = st.slider('Price: £', round(asos['price'].min()),round(asos['price'].max()), key='4')
    
    
    return producttype, brand, colour, price


def ASOS_trendy():

    ######## Import libraries ########
    import pandas as pd
    import numpy as np
    from random import randint
    import pickle
    import streamlit as st
    ######## Import data ######## 
    
    asos = pd.read_csv('data/asos_clusters.csv')
    
    
    ######## Valid input options ########
   
    valid_producttype = list(asos['#search'].unique())
    valid_brand = list(asos['#brand'].unique())
    valid_colour = list(asos['#colour'].unique())

    
    ######## User input ######## 
    producttype = st.selectbox('Product Type: ', valid_producttype, key='1')
    brand  = st.selectbox('Brand: ', valid_brand, key='2')
    colour = st.selectbox('Colour: ', valid_colour, key='3')
    price  = st.number_input('Price: £', round(asos['price'].min()),round(asos['price'].max()), key='4')
     
    if st.button("LETS GO!!"):
        ######## Create a DF ######## 
        input_df = pd.DataFrame({'price': [price],'#search': [producttype], '#brand': [brand] ,'#colour': [colour]}).reset_index(drop=True)
        
        dict_cat = dict(zip(asos['#search'],asos['categories']))
        input_df['categories'] = dict_cat[producttype]
        input_df = input_df[['price', '#search', '#brand', 'categories' ,'#colour']]
        
        input_df['price'] = input_df['price'].astype(float)
        
        ######### Loading the model and scalers from the file using pickle ########
    
        model = pickle.load(open('models/regression_model.pkl','rb'))
        model_ohe = pickle.load(open('encoders/regression_ohe.pkl','rb'))
        model_minmax = pickle.load(open('transformers/regression_minmax.pkl','rb'))
        
        ###### USING OUR MODEL ######
    
        X_num_r = input_df[['price']]
        X_cat_r = input_df.drop(columns=['price'],axis =1)
        
        X_num_r_tf = model_minmax.transform(X_num_r)
        X_num_r_tf = pd.DataFrame(X_num_r_tf,columns=X_num_r.columns)
        
        X_cat_r_ohe = model_ohe.transform(X_cat_r).toarray()
        cols_r = model_ohe.get_feature_names_out(input_features=X_cat_r.columns)
        X_cat_r_t = pd.DataFrame(X_cat_r_ohe, columns=cols_r)
        
        #Concat 
    
        X_r_treated = pd.concat([X_num_r_tf, X_cat_r_t], axis = 1)
        
        ##### PREDICT ####
    
        
    
        if model.predict(X_r_treated)[0] > np.quantile(asos['potential_hashtag'], q=0.65):
            st.markdown(" ")
            st.markdown("##### That's really trendy!")
        else:
            model = pickle.load(open('models/clustering_model.pkl','rb'))
        
        
            input_df['cluster'] = model.predict(X_r_treated)
        
        
            pd.set_option('display.max_colwidth', None)
            st.markdown(" ")
            st.markdown("###### What you input is not very trendy, we found a similar, more trendy product. Voila:")
            st.markdown("https://www.google.com/search?q="+asos[(asos['cluster'] == model.predict(X_r_treated)[0])&(asos['trendy'] == 'High')].reset_index().loc[[randint(0,len(asos[(asos['cluster'] == model.predict(X_r_treated)[0])&(asos['trendy'] == 'High')]))]]['name'].str.replace(' ','%20').to_string(index = False))
            