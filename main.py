import pandas as pd
import numpy as np
from mip import Model, xsum, maximize, BINARY
import streamlit as st


def main():
    full_menu = pd.read_csv('kiwami_menu.csv')
    full_menu.fillna("", inplace=True)
    df = full_menu.loc[full_menu.category != 'Extra']

    st.sidebar.title('Kiwami Sushi Ramen')
    T = st.sidebar.number_input('How much do you want to spend?', min_value=1, value=20)
    F = st.sidebar.number_input('How many people are you feeding?', min_value=1, max_value=20, value=2)

    if st.button('Generate Order!'):
        selected = order_generator(T, F, df)
        st.table(df.iloc[selected])


def order_generator(T, F, df):
    I = range(df.shape[0])
    m = Model("menu")
    x = [m.add_var(var_type=BINARY) for i in I]
    p = np.array(df.price)
    r = np.array(np.random.rand(df.shape[0]))
    f = np.array(df.fullness)
    m.objective = maximize(xsum(p[i] * x[i] * r[i] for i in I))
    m += xsum(p[i] * x[i] for i in I) <= T * (1 + np.random.rand() * 0.05)
    m += xsum(f[i] for i in I) >= F * 0.93
    m += xsum(f[i] for i in I) <= 2
    m.optimize()
    selected = [i for i in I if x[i].x >= 0.99]
    return selected


if __name__ == '__main__':
    main()
