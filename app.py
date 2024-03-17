import streamlit as st
import pandas as pd
# Library for matrix operations
import numpy as np

def HomePage():
    st.title('Welcome to Matrix Solver')
    st.write('This is a web app that will help you solve a system of linear equations using the matrix method.')
    st.write('Please select "Matrix Solver" from the navigation to start solving your system of linear equations.')
    
    st.title('Matrix Solver Web App')
    st.write("Welcome to the Matrix Solver Web App! This app will help you solve a system of linear equations using the matrix method.")

    st.write("Here's a simple example of a system of linear equations:")
    st.latex(r'''
    \begin{align*}
    x + y + z &= 6 \\
    y + z &= -4 \\
    z &= 3
    \end{align*}
    ''')

    st.write("This system of linear equations can be represented as a matrix equation:")
    st.latex(r'''
    \begin{bmatrix}
    1 & 1 & 1 \\
    0 & 1 & 1 \\
    0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    x \\
    y \\
    z
    \end{bmatrix}
    =
    \begin{bmatrix}
    6 \\
    -4 \\
    3
    \end{bmatrix}
    ''')
            

    st.write("Here's our first attempt at using data to create a table:")
    st.write(pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    }))

def MatrixSolverPage():
    st.title('Matrix Solver')
    st.write('This is the Matrix Solver page.')
    
    col1, col2 = st.columns(2)
    with col1:
        # Input fields for the size of the matrix
        m = st.number_input('Enter the number of rows', min_value=1, value=2)
    with col2:
        n = st.number_input('Enter the number of columns', min_value=1, value=2)
    # Generate button
    if st.button('Generate Matrix'):
        with col1: # For Matrix A Input
            st.write('Enter the elements of Matrix A')
            A = []
            # Create subcolumns for each element of the matrix
            for i in range(m):
                subcolumns = []
                for j in range(n):
                    subcolumns = st.columns(n)
                    with subcolumns[j]:
                        A.append(st.number_input(f'A[{i+1},{j+1}]', value=0, key=f'A[{i+1},{j+1}]'))


def AboutPage():
    st.title('About')
    st.write('This is the about page.')

# Sidebar
st.sidebar.title('Matrix Solver')
st.sidebar.write('Vector and Matrix Theory Project.')
# Sidebar Navigation to Home
nav = st.sidebar.radio('', ['Home', 'Matrix Solver', 'About'])

if nav == 'Home':
    HomePage()
elif nav == 'Matrix Solver':
    MatrixSolverPage()
else:
    AboutPage()
