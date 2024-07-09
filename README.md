# Grading_Predictions

This program is designed to take in the border measurements and condition of the card and output a predicted grade. A blank csv sheet is provided with headers to input your cards features. 

The best set up is to make a file location that will hold 4 items: the .exe file, the data you want processed, the data used to train the model and the output file.
Prior to running the .exe, you muct evaluate your cards and populate a csv accurately. 


Follow the directions below to accurately rate the card. 

Border Measurements:
Measurements should be taken using digital calipers and using milimeter unit of measure. Only the front borders get measured. 
EXCEPTION: if the card's art removes the boarder on a side, the back border meaurements will take the place of the front borders.

Corners:
Each corner gets one 'vote'. Evaluate the corner of the card and pick the worst category that defines the corner. Place a 1 in that location. If a number already exists, add 1 to it. For example if a card has perfect corners, a 4 should be placed under "clean cut". If a card has one corner bent, one with a small white tick and the others are perfect, there should be a 2 under "clean cut", 1 under "bent", and 1 under "white tick". 

Edges:
Evaluate each edge, both front and back. The 'voting' system is similar to method used for corners where you will add a 1 in the feature label that represents that edge.

Surface:
The surface evaluation is slightly different. A 1 should be placed to represent the presence of that feature if it exists on the card, front or back. For example, a perfect card should have a 1 under "Wax is Mint" and nothing under the remainder of options. If a card has bends, creases, and noticible scratches, a 1 shold be placed under each of those categories. 




After rating all your cards, you may save the file in the same directory(folder) as the python exe script as 'self_grade_sheet.csv' or you may choose a custom file location and name. Remember this location and name as the program will ask for the file path to retrieve the information. 

Run the .exe file. A new csv file should be placed in you choosen location named "Predicted_Grades.csv" and contains the predictions. Happy Grading!!
