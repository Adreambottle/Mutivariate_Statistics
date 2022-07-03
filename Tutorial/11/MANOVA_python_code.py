import pandas as pd
from statsmodels.multivariate.manova import MANOVA

# Download the data
url = 'https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv'
df = pd.read_csv(url, index_col=0)
df.columns = df.columns.str.replace(".", "_")
df.columns

# View the data
df.head()


# Perform the MANOVA
maov = MANOVA.from_formula('Sepal_Length + Sepal_Width + \
                            Petal_Length + Petal_Width  ~ Species', data=df)

# View the summary of MANOVA
print(maov.mv_test())
