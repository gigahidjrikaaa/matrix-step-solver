import streamlit as st
import pandas as pd
import numpy as np

def HomePage():
    st.title('Matrix Solver Web App')
    st.write("Welcome to the Matrix Solver Web App! This app will help you solve a system of linear equations using the matrix method.")
    st.info('Please select "Matrix Solver" from the navigation to start solving your own system of linear equations.')

    st.divider()
    st.header('Introduction')

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
            
    # Create a matrix using numpy
    A = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    b = np.array([6, -4, 3])
    matrix_A = pd.DataFrame(A, columns=['x', 'y', 'z'])
    matrix_b = pd.DataFrame(b, columns=['Constants'])
    x = np.linalg.solve(A, b)
    st.write("The solution to the system of linear equations is:")
    st.write(x)
    st.latex(r'''
    \begin{bmatrix}
    x \\
    y \\
    z
    \end{bmatrix}
    =
    \begin{bmatrix}
    1 \\
    -7 \\
    3
    \end{bmatrix}
    ''')

    # Display Matrix A without the row header
    st.write('Matrix A:')
    st.write(matrix_A, header=False)

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
    st.write('This is a project made for the _Vector and Matrix Theory_ course under **Prof. Dr.rer.nat. Indah Emilia Wijayanti, S.Si., M.Si.** at the Department of Mathematics, Faculty of Mathematics and Natural Sciences, Universitas Gadjah Mada.')
    st.divider()
    st.write('This project is made by:')
    st.write('1. Giga Hidjrika Aura Adkhy - 21/479228/TK/52833')
    st.write('2. ')
    st.write('3. ')    



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
