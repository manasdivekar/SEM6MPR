import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

team_images = {
    'Sunrisers Hyderabad': 'https://media.crictracker.com/media/attachments/1669546081953_Sunrisers-Hyderabad.jpeg',
    'Mumbai Indians': 'https://media.crictracker.com/media/attachments/1680698558309_Mumbai-Indians-new-logo.jpeg',
    'Royal Challengers Bangalore': 'https://media.crictracker.com/media/attachments/1674557817477_Royal-Challengers-Bangalore.jpeg',
    'Kolkata Knight Riders': 'https://media.crictracker.com/media/attachments/1671612736819_KKR-logo.jpeg',
    'Kings XI Punjab': 'https://media.crictracker.com/media/attachments/1671443590572_Punjab-Kings.jpeg',
    'Chennai Super Kings': 'https://media.crictracker.com/media/attachments/1674543242945_CSK-Logo.jpeg',
    'Rajasthan Royals': 'https://media.crictracker.com/media/attachments/1674647344276_Rajasthan-Royals-Logo.jpeg',
    'Delhi Capitals': 'https://media.crictracker.com/media/featureimage/Delhi-Capitals-logo-1.jpg'
}

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe_svm.pkl', 'rb'))
st.title('VICTORY VISION')



col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
    if batting_team in team_images:
        st.image(team_images[batting_team], use_column_width=True)

with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))
    if bowling_team in team_images:
        st.image(team_images[bowling_team], use_column_width=True)

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input(f'Target Set by {bowling_team}')

col4, col5, col6 = st.columns(3)

with col4:
    score = st.number_input(f'Score Of {batting_team}')
with col5:
    overs = st.number_input('Overs completed')
with col6:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets],
                             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str((win * 100)) + "%")
    st.header(bowling_team + "- " + str((loss * 100)) + "%")
