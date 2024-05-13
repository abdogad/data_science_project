#!/usr/bin/env python
# coding: utf-8


import pandas as pd
def vis_plot1(train):
                import plotly.graph_objects as go

                x=train.iloc[:,-1].value_counts().to_dict()
                fig = go.Figure(data=[go.Bar(x=list(x.keys()), y=list(x.values()))])

                # Customize the layout
                fig.update_layout(
                    title='Class Distribution',
                    xaxis=dict(title='Class'),
                    yaxis=dict(title='Number of Samples'),
                    showlegend=False,
                )

                # Show the plot
                fig.show()





def vis_plot2(train):
                import pandas as pd
                import plotly.express as px

                numerical_cols = train.select_dtypes(include=['int', 'float']).columns

                for col in numerical_cols[:4]:
                    fig = px.histogram(
                        train,  # Dataframe containing your data
                        x=col,  # Column name for the x-axis
                        color='CLASS',  # Column name for coloring by class
                    )

                    # Customize appearance as needed (examples):
                    fig.update_layout(
                        xaxis_title_font_size=12,  # Adjust font size
                        yaxis_title_font_size=12,
                        legend_title_text="Class",  # Set legend title
                        legend_title_font_size=12,
                        plot_bgcolor="white"  # Set background color
                    )

                    fig.update_traces(marker_line_color="black", marker_line_width=0.5)  # Enhance marker visibility

                    fig.show()





def vis_plot3(train):
    # Visualize relationship between numerical variables (scatter plot matrix)
        numerical_cols = train.select_dtypes(include=['int', 'float']).columns

        import matplotlib.pyplot as plt
        pd.plotting.scatter_matrix(train[numerical_cols].iloc[:,:20], figsize=(20, 20), diagonal='kde')
        plt.show()





def vis_plot4(train):
                import pandas as pd
                import plotly.express as px

                # Assuming `train` is your DataFrame containing numerical columns

                numerical_cols = train.select_dtypes(include=['int', 'float']).columns

                # Create a scatter matrix with a larger figure size
                fig = px.scatter_matrix(
                    train,
                    dimensions=numerical_cols,
                    color='CLASS',
                    title='Scatter Matrix',
                    width=1000,  # Adjust width for desired image size
                    height=1000   # Adjust height for desired image size
                )

                # Customize layout for clarity (optional)
                fig.update_layout(
                    title_x=0.5,  # Center title horizontally
                    margin=dict(l=50, r=50, b=50, t=50)  # Adjust margins for better spacing
                )

                fig.show()




def vis_plot5(train):
        import plotly.express as px


        categorical_cols = train.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            fig = px.pie(train, names=col, title=f'Pie Chart: {col}')
            fig.show()





def vis_plot6(train):
    import pandas as pd
    import plotly.graph_objects as go

    numerical_cols = train.select_dtypes(include=['int', 'float']).columns

    # Calculate correlation matrix
    correlation_matrix = train[numerical_cols].corr()

    # Create heatmap with adjustable figure size
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        colorbar=dict(title='Correlation')
    ))

    # Set desired figure size (adjust width and height as needed)
    fig.update_layout(
        title='Correlation Heatmap of Numerical Features',
        xaxis=dict(title='Features'),
        yaxis=dict(title='Features'),
        width=1000,  # Adjust width for desired image size
        height=1000   # Adjust height for desired image size
    )

    fig.show()





def vis_plot7(train):
                import pandas as pd
                import plotly.express as px


                numerical_cols = train.select_dtypes(include=['int', 'float']).columns


                fig = px.area(train,  y=numerical_cols,color='CLASS')
                fig.show()







