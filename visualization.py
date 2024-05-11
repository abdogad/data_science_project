# **visualization**
import plotly.io as pio
pio.renderers.default = "browser"
#import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


#histogram
import pandas as pd
import plotly.express as px


numerical_cols = train.select_dtypes(include=['int', 'float']).columns

for col in numerical_cols:
    fig = px.histogram(train, x=col, color='CLASS', 
                       title=f'Histogram: {col}', 
                       labels={col: col.capitalize()})
    fig.show()



#scatter plot matrix
import pandas as pd
import plotly.express as px



columns = ['MINOR_AXIS', 'ECCENTRICITY', 'ASPECT_RATIO', 'ROUNDNESS', 'COMPACTNESS', 
           'SHAPEFACTOR_2', 'SHAPEFACTOR_3', 'entropyRR', 'entropyB', 'entropyCr']

# Creating scatter matrix using Plotly
scatter_matrix = px.scatter_matrix(train, dimensions=columns, color='CLASS', labels={col:col for col in columns})
scatter_matrix.update_layout(
    width=1000,  # Adjust width as needed
    height=1000,  # Adjust height as needed
)
# Update the layout to adjust the font size of the labels
scatter_matrix.update_layout(font=dict(size=5))
scatter_matrix.show()




#scatter
import pandas as pd
import plotly.express as px


numerical_cols = train.select_dtypes(include=['int', 'float']).columns

for col_x in numerical_cols:
    for col_y in numerical_cols:
        if col_x != col_y:  # To avoid plotting the same column against itself
            fig = px.scatter(train, x=col_x, y=col_y, color='CLASS',
                             title=f'Scatter Plot: {col_x} vs {col_y}')
            fig.show()





#Bar Chart
import pandas as pd
import plotly.express as px


categorical_cols = train.select_dtypes(include=['object']).columns

for col in categorical_cols:
    fig = px.bar(train[col].value_counts(), title=f'Bar Chart: {col}')
    fig.show()




#Box Plot
import pandas as pd
import numpy as np
import plotly.express as px

# Define a function to remove outliers from each class separately
def remove_outliers_by_class(df, col, class_col):
    df_no_outliers = pd.DataFrame()
    for class_label in df[class_col].unique():
        df_class = df[df[class_col] == class_label]
        q1 = df_class[col].quantile(0.25)
        q3 = df_class[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_no_outliers = pd.concat([df_no_outliers, df_class[(df_class[col] >= lower_bound) & (df_class[col] <= upper_bound)]])
    return df_no_outliers


numerical_cols = train.select_dtypes(include=['int', 'float']).columns

for col in numerical_cols:
    # Remove outliers from each class separately
    train_no_outliers = remove_outliers_by_class(train, col, 'CLASS')
    
    # Create box plot without outliers
    fig = px.box(train_no_outliers, y=col, color='CLASS', 
                 title=f'Box Plot without Outliers by Class: {col}', 
                 labels={col: col.capitalize()})
    fig.show()
import pandas as pd
import plotly.express as px
# Define a function to remove outliers from each class separately
def remove_outliers_by_class(df, col, class_col):
    df_no_outliers = pd.DataFrame()
    for class_label in df[class_col].unique():
        df_class = df[df[class_col] == class_label]
        q1 = df_class[col].quantile(0.25)
        q3 = df_class[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_no_outliers = pd.concat([df_no_outliers, df_class[(df_class[col] >= lower_bound) & (df_class[col] <= upper_bound)]])
    return df_no_outliers


numerical_cols = train.select_dtypes(include=['int', 'float']).columns

for col in numerical_cols:
     # Remove outliers from each class separately
    train = remove_outliers_by_class(train, col, 'CLASS')
    
    fig = px.box(train, y=col, title=f'Box Plot: {col}')
    fig.show()
import plotly.express as px


fig = px.box(train, 
             y=['MINOR_AXIS', 'ECCENTRICITY', 'ASPECT_RATIO', 'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_2', 'SHAPEFACTOR_3', 'entropyRR', 'entropyB', 'entropyCr'], 
             color='CLASS',
             title='Box Plot for Various Attributes',
             labels={'value': 'Attribute Value', 'CLASS': 'Class'},
             template='plotly_dark'
            )

fig.show()




#bubble
import plotly.express as px


numerical_cols = train.select_dtypes(include=['int', 'float']).columns

for i, col_x in enumerate(numerical_cols):
    for j, col_y in enumerate(numerical_cols):
        if i != j:  # To avoid plotting the same column against itself
            fig = px.scatter(train, 
                             x=col_x, 
                             y=col_y,  
                             color='CLASS', 
                             hover_name='CLASS', 
                             title=f'Bubble Chart: {col_x} vs {col_y}',
                             labels={col_x: col_x.capitalize(), col_y: col_y.capitalize(), 'CLASS': 'Class'},
                             template='plotly_dark' 
                            )
            fig.show()
import pandas as pd
import plotly.express as px


numerical_cols = train.select_dtypes(include=['int', 'float']).columns

for col in numerical_cols:
    fig = px.scatter(train, x=train.index, y=col, 
                 color='CLASS',  title=f'Bubble Chart: {col}')
    fig.show()




#line
import plotly.express as px


fig = px.line(train, 
              x=train.index,  
              y=['MINOR_AXIS', 'ECCENTRICITY', 'ASPECT_RATIO', 'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_2', 'SHAPEFACTOR_3', 'entropyRR', 'entropyB', 'entropyCr'], 
              color='CLASS',
              title='Line Chart: Various Attributes over Time',
              labels={'value': 'Attribute Value', 'CLASS': 'Class'},
              template='plotly_dark',
              line_dash='CLASS'  # Use different line styles for each class
             )

fig.show()





#area
import plotly.express as px


fig = px.area(train, 
              x=train.index,  
              y=['MINOR_AXIS', 'ECCENTRICITY', 'ASPECT_RATIO', 'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_2', 'SHAPEFACTOR_3', 'entropyRR', 'entropyB', 'entropyCr'], 
              color='CLASS',
              title='Area Chart: Various Attributes over Time',
              labels={'value': 'Attribute Value', 'CLASS': 'Class'},
              template='plotly_dark'
             )

fig.show()
import pandas as pd
import plotly.express as px


numerical_cols = train.select_dtypes(include=['int', 'float']).columns

for col in numerical_cols:
    fig = px.area(train, x=train.index, y=col,color='CLASS', title=f'Area Chart: {col}')
    fig.show()





#Pie
import plotly.express as px


categorical_cols = train.select_dtypes(include=['object']).columns

for col in categorical_cols:
    fig = px.pie(train, names=col, title=f'Pie Chart: {col}')
    fig.show()
import plotly.express as px

numerical_cols = train.select_dtypes(include=['int', 'float']).columns

for col in numerical_cols:
    # Group by the numerical column and count occurrences of each class
    class_counts = train.groupby(col)['CLASS'].value_counts().unstack(fill_value=0)
    
    # Create pie chart for each numerical column
    for class_col in class_counts.columns:
        fig = px.pie(values=class_counts[class_col], names=class_counts.index, title=f'Pie Chart: {class_col} by CLASS')
        fig.show()





#Violin plots
import plotly.express as px


numerical_cols = train.select_dtypes(include=['int', 'float']).columns

for col in numerical_cols:
    fig = px.violin(train, y=col, box=True, points="all", color='CLASS', 
                    title=f'Violin Plot: {col}', 
                    labels={col: col.capitalize(), 'CLASS': 'Class'})
    fig.show()
import plotly.express as px


fig = px.violin(train, 
                y=['MINOR_AXIS', 'ECCENTRICITY', 'ASPECT_RATIO', 'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_2', 'SHAPEFACTOR_3', 'entropyRR', 'entropyB', 'entropyCr'], 
                box=True,  # Include a box plot inside the violin
                points="all",  # Show all data points
                color='CLASS',  # Color by class
                title='Violin Plot for Various Attributes',
                labels={'value': 'Attribute Value', 'CLASS': 'Class'},
                template='plotly_dark'
               )

fig.show()




#heatmap
import pandas as pd
import plotly.graph_objects as go


numerical_cols = train.select_dtypes(include=['int', 'float']).columns

# Calculate correlation matrix
correlation_matrix = train[numerical_cols].corr()

# Create heatmap
fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        colorbar=dict(title='Correlation')
))

fig.update_layout(
    title='Correlation Heatmap of Numerical Features',
    xaxis=dict(title='Features'),
    yaxis=dict(title='Features')
)

fig.show()

