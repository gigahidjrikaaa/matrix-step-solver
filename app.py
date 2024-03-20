import streamlit as st
import pandas as pd
import numpy as np
from fractions import Fraction

######################
# Function to write pandas dataframe in latex
######################
def df_to_latex(df):
    # Convert the dataframe to a string in latex format
    latex = df.to_latex(index=False, escape=False)
    # Replace the \n with \\ to make it a single line
    latex = latex.replace('\n', '\\\\\n')
    # Add the latex environment
    latex = r'\begin{bmatrix}' + '\n' + latex + r'\end{bmatrix}'
    return latex

######################
# Function for matrix number 1
######################
def display_matrix(matrix):
    df = pd.DataFrame(matrix)
    st.dataframe(df)

# Function for Gaussian elimination
def gauss_elimination(matrix):
    n = len(matrix)
    for i in range(n):
        # Pivoting
        if matrix[i][i] == 0:
            for j in range(i+1, n):
                if matrix[j][i] != 0:
                    matrix[i], matrix[j] = matrix[j], matrix[i]
                    st.write("Swap row", i+1, "and row", j+1)
                    display_matrix(matrix)
                    break
        # Elimination
        for j in range(i+1, n):
            if matrix[j][i] != 0:
                factor = matrix[j][i] / matrix[i][i]
                for k in range(i, n):
                    matrix[j][k] -= factor * matrix[i][k]
                st.write("\nEliminate row", j+1, "using row", i+1, "with factor", Fraction(factor).limit_denominator())
                display_matrix(matrix)
    return matrix

# Function to calculate determinant
def determinant(matrix):
    det = 1
    for i in range(len(matrix)):
        det *= matrix[i][i]
    return det

######################
# Home Page Section
# latex of the example linear equations written here
######################
def HomePage():
    st.title('Matrix Solver Web App')
    st.write("Welcome to the Matrix Solver Web App! This app will help you solve a system of linear equations using the matrix method.")
    st.info('Please select "Matrix Solver" from the navigation to start solving your own system of linear equations.')

    st.divider()
    st.header('Introduction')

    ### Matrix number 1 ###
    # Displaying the matrix
    st.write("Matrix:")
    matrix = [
        [2, 3, 1, 2],
        [-4, 1, -1, 0],
        [0, 1, 2, 1],
        [0, 0, 1, 0]
    ]
    display_matrix(matrix)

    # Applying Gauss elimination
    st.write("\nApplying Gaussian Elimination:")
    gauss_matrix = gauss_elimination(matrix)
    # Calculating determinant
    det = determinant(gauss_matrix)
    # Displaying the matrix after Gaussian elimination and determinant
    st.write("\nMatrix after Gauss elimination:")
    display_matrix(gauss_matrix)
    st.write("\nDeterminant:", det)

    ### CONTOH ###
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
    matrix_A = pd.DataFrame(A)
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

    # Matrix using dataframe
    st.write('Matrix using dataframe:')

    # Test data editor
    matrix_A = st.data_editor(matrix_A, hide_index=True, key='matrix_A')
    st.data_editor(matrix_b)

    # When matrix_A is edited, the value of determinant will be updated
    if st.button('Calculate Determinant'):
        st.write('Determinant of matrix A:', np.linalg.det(matrix_A.values))


######################
# Matrix Solver Section
# Page for custom single matrix solver.
######################
def MatrixSolverPage():
    st.title('Matrix Solver')
    st.write('This is the Matrix Solver page.')
    st.write('This page will help you solve a system of linear equations using the matrix method.')
    st.divider()

    st.subheader('Matrix Input')
    col = st.columns(2)
    with col[0]:
        st.number_input('Number of rows (m)', min_value=1, value=2, key='m')
    with col[1]:
        st.number_input('Number of columns (n)', min_value=1, value=2, key='n')

    m = st.number_input('Number of rows (m)', min_value=1, value=2, key='m')
    n = st.number_input('Number of columns (n)', min_value=1, value=2, key='n')
    A = []
    subcolumns = []
    subcolumns = st.columns(n)
    for i in range(m):
        for j in range(n):
            with subcolumns[j]:
                A.append(st.number_input(f'A[{i+1},{j+1}]', value=0, key=f'A[{i+1},{j+1}]'))

    # Create a button to get the determinant step by step
    if st.button('Find Determinant'):
        A = np.array(A).reshape(m, n)
        determinant = np.linalg.det(A)
        st.write('The determinant of the matrix is:', determinant)

    # Create a button to solve the matrix
    if st.button('Solve'):
        A = np.array(A).reshape(m, n)
        b = np.array([6, -4, 3])
        x = np.linalg.solve(A, b)
        st.write('The solution to the system of linear equations is:')
        st.write(x)
        # Write the solution in latex
        X_latex = r'\begin{bmatrix}'
        for i in range(x.shape[0]): 
            X_latex += str(x[i])
            if i < x.shape[0] - 1:
                X_latex += r' \\ '
        X_latex += r'\end{bmatrix}'
        st.latex(X_latex)

            
######################
# Operations for the Matrices
# A and B are the matrices
######################
def Addition(A, B):
    A = np.array(A).reshape(len(A)//2, 2)
    B = np.array(B).reshape(len(B)//2, 2)
    return A + B

def Subtraction(A, B):
    A = np.array(A).reshape(len(A)//2, 2)
    B = np.array(B).reshape(len(B)//2, 2)
    return A - B

def Multiplication(A, B):
    A = np.array(A).reshape(len(A)//2, 2)
    B = np.array(B).reshape(len(B)//2, 2)
    return A @ B

######################
# Two Matrix Operations Section
# Page for matrix operations
######################
def TwoMatrixOperationsPage():
    st.title('Two Matrix Operations')
    st.write('This is the Two Matrix Operations page.')
    st.write('This page will help you perform operations on two matrices.')
    st.write('Please select the operation you want to perform from the dropdown menu.')

    col = st.columns(3)
    with col[0]:
        # Matrix A Input
        st.subheader('Matrix A')
        m = st.number_input('Number of rows (m)', min_value=1, value=2, key='m')
        n = st.number_input('Number of columns (n)', min_value=1, value=2, key='n')
        A = []
        subcolumns = []
        subcolumns = st.columns(n)
        for i in range(m):
            for j in range(n):
                with subcolumns[j]:
                    A.append(st.number_input(f'A[{i+1},{j+1}]', value=0, key=f'A[{i+1},{j+1}]'))
        
    with col[2]:
        # Matrix B Input
        st.subheader('Matrix B')
        p = st.number_input('Number of rows (p)', min_value=1, value=2, key='p')
        q = st.number_input('Number of columns (q)', min_value=1, value=2, key='q')
        B = []
        subcolumns = []
        subcolumns = st.columns(q)
        for i in range(p):
            for j in range(q):
                with subcolumns[j]:
                    B.append(st.number_input(f'B[{i+1},{j+1}]', value=0, key=f'B[{i+1},{j+1}]'))

    with col[1]:
        if st.button('Addition'):
            st.latex(df_to_latex(pd.DataFrame(Addition(A, B))))
        if st.button('Subtraction'):
            st.latex(df_to_latex(pd.DataFrame(Subtraction(A, B))))
        if st.button('Multiplication'):
            st.latex(df_to_latex(pd.DataFrame(Multiplication(A, B))))
    
    # Create a button to solve the matrix
    if st.button('Solve'):
        A = np.array(A).reshape(m, n)
        B = np.array(B).reshape(p, q)
        if n != p:
            st.error('The number of columns of matrix A must be equal to the number of rows of matrix B.')
        else:
            X = np.linalg.solve(A, B)
            st.write('The solution to the system of linear equations is:')
            st.write(X)
            # Write the solution in latex
            X_latex = r'\begin{bmatrix}'
            for i in range(X.shape[0]): 
                for j in range(X.shape[1]):
                    X_latex += str(X[i, j])
                    if j < X.shape[1] - 1:
                        X_latex += ' & '
                    else:
                        X_latex += r' \\ '
            X_latex += r'\end{bmatrix}'
            st.latex(X_latex)       

######################
# About Page Section
######################
def AboutPage():
    st.title('About')
    st.write('This is a project made for the _Vector and Matrix Theory_ course under **Prof. Dr.rer.nat. Indah Emilia Wijayanti, S.Si., M.Si.** at the Department of Mathematics, Faculty of Mathematics and Natural Sciences, Universitas Gadjah Mada.')
    st.divider()
    st.write('This project is made by:')
    st.write('1. Giga Hidjrika Aura Adkhy - 21/479228/TK/52833')
    st.write('2. Salwa Maharani - 21/481194/TK/53113')
    st.write('3. Noer Azizah PS - 20/463611/TK/51603')    

######################
# Sidebar
######################
st.sidebar.title('Matrix Solver')
st.sidebar.write('Vector and Matrix Theory Project.')
# Sidebar Navigation to Home
nav = st.sidebar.radio('', ['Home', 'Single Matrix Solver', 'Two Matrix Operations', 'About'])


######################
# Main Section
######################
if nav == 'Home':
    HomePage()
elif nav == 'Single Matrix Solver':
    MatrixSolverPage()
elif nav == 'Two Matrix Operations':
    TwoMatrixOperationsPage()
else:
    AboutPage()
