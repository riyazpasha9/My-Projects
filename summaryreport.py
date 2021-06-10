import pandas as pd

from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList



df = pd.read_csv('Y:\\ACOE Project DOCS\\Team Members Folder\\Riyaz\\Trackers\\JIRA dump for Q4.csv',usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print(df.head())

#input_month = input("Enter the Month to generate the report:")
#print(input_month)

# Section 3 - Testing Pivot Tables
#filtered = df[df['Month'] == input_month]
filtered = df
monthly_volumes = pd.pivot_table(filtered, index = ['Assignee','Status'], columns = filtered['Reporter'] , values = 'Priority', aggfunc='count')

print("Monthly volumes Pivot Table:")
print(monthly_volumes.head())

#Section 04 - Creating and Excel Workbook
file_path = 'Y:\\Project DOCS\\Team Members Folder\\Riyaz\\Trackers\\Monthly.xlsx'
monthly_volumes.to_excel(file_path, sheet_name = 'Pivot Volumes', startrow=3)


# Section 05 - Loading the Workbook
wb = load_workbook(file_path)
sheet1 = wb['Pivot Volumes']

# Section 06 - Formatting the First Sheet
sheet1['A1'] = 'Monthly Volumes'
sheet1['A2'] = 'Month wise'

sheet1['A1'].style = 'Title'
sheet1['A2'].style = 'Headline 2'


# Section 07 - Adding a Bar Chart
bar_chart = BarChart()
bar_chart.type = "col"
bar_chart.style = 10
bar_chart.y_axis.title = 'Volumes'
bar_chart.x_axis.title = 'Status'
data = Reference(sheet1, min_col=2, max_col=4, min_row=4, max_row=8)
categories = Reference(sheet1, min_col=1, max_col=1, min_row=5, max_row=8)
bar_chart.add_data(data, titles_from_data=True)
bar_chart.set_categories(categories)
bar_chart.dataLabels = DataLabelList()
bar_chart.dataLabels.showVal = True
sheet1.add_chart(bar_chart, "F4")

bar_chart.title = 'Month Wise Volumes of Requests'
bar_chart.style = 3
wb.save(filename = file_path)
