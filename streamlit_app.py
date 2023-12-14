import streamlit as st

st.title('Welcome to Our App')
st.write('This is the home page of our application. You can choose to test the model or see our work.')

if st.button('Test Model'):
    # Code to handle the 'Test Model' action
    st.write('You clicked to test the model!')

if st.button('See Our Work'):
    # Code to handle the 'See Our Work' action
    st.write('You clicked to see our work!')
