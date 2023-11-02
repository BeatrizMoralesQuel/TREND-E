###############################
# This program lets you       #
# - Create a dashboard        #
# - Evevry dashboard page is  #
# created in a separate file  #
###############################

# Python libraries
import streamlit as st
from PIL import Image
from recommender import ASOS_trendy, user_input

def main():

    #############
    # Main page #
    #############
    
     ######## Selecting what is going to be in our app menu ########

    options = ['Home', 'Trendy or Not?']
    choice = st.sidebar.selectbox("Menu",options) #key means that if I had multiple input methods on one page, they all should have a different key
    
    
    
     ######## Saying to this menu, what it has to open in every case ########


    if ( choice == 'Home' ):
        st.markdown("<h1 style='text-align: center; color: black;'>TREND-E</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: gray;'>Welcome to the trendy fashion product recommender</h4>", unsafe_allow_html=True)
        st.image("./images/Fashion-Jackson-Instagram-3.jpg")
        col1, col2, col3 = st.columns(3)

        #with col1:
        #    st.write('', width = 1)
#
        #with col2:
        #    st.image("./images/Fashion-Jackson-Instagram-3.jpg")

       # with col3:
       #     st.write('', width = 1)
        st.markdown("<h5 style='text-align: center; color: black;'>Let us help you choose what to produce or add to your stock in just a few minutes</h5>", unsafe_allow_html=True)
        pass
    
   
    
    
    elif ( choice == 'Trendy or Not?' ):
        ASOS_trendy()
        pass

    else:
        st.stop()
    
    


main()
