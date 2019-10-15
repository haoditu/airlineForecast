import pandas as pd


def airlineForecast(validationDataFileName,trainingDataFileName):
    
    tdt = pd.read_csv(trainingDataFileName)
    vdt = pd.read_csv(validationDataFileName)
                
    def mul_model_dow(validationDataFileName,trainingDataFileName):
        tdt = trainingDataFileName
        vdt = validationDataFileName
    
        """Multiplicative model"""
        #trainning data  
        # days prior = depature - book and day of week in training data
        tdt['days prior'] = (pd.to_datetime(tdt['departure_date']) - pd.to_datetime(tdt['booking_date'])).dt.days
        departure_date = pd.to_datetime(tdt['departure_date'])
        tdt['day_of_week']= departure_date.dt.weekday_name
        
        #total demand of tickets for each departure day
        tdt['final'] = tdt.loc[tdt['days prior'] == 0,'cum_bookings']
        final = tdt.loc[tdt['days prior'] == 0, ['departure_date', 'final']]
        tdt = tdt.merge(final, left_on =['departure_date'], right_on = ['departure_date'])
        tdt = tdt.drop('final_x', axis = 1)
        tdt = tdt.rename(columns = {'final_y': 'final_demanding'})
    
        #booking rate
        tdt['booking_rate'] = tdt['cum_bookings']/tdt['final_demanding']
        """  day of week """        
        # avg booking rate by same departure day of week 
        dow_avg_booking = tdt.groupby(['days prior', 'day_of_week']).mean()
        dow_avg_booking = dow_avg_booking.drop(['cum_bookings', 'final_demanding'], axis=1).reset_index()
        dow_avg_booking = dow_avg_booking.iloc[0:203]
        #validation data
        vdt['days prior'] = (pd.to_datetime(vdt['departure_date']) - pd.to_datetime(vdt['booking_date'])).dt.days
        vdt['day_of_week']= pd.to_datetime(vdt['departure_date']).dt.weekday_name
        
        #merge new dt with validation data
        df = vdt.merge(dow_avg_booking, how = 'left' , left_on = ['days prior', 'day_of_week'], right_on = ['days prior', 'day_of_week'])
        df = df.loc[df["days prior"]!= 0]
        
        #calculating MASE 
        df['dow_forecast'] = df['cum_bookings']/df['booking_rate']
        df['dow_error_multiplicative'] =abs(df['final_demand'] - df['dow_forecast'])
        df['error_naive'] = abs(df['final_demand'] - df['naive_forecast'])
        
        MASE_dow = df['dow_error_multiplicative'].sum() / df['error_naive'].sum()
        
        # forecast
        forecasts_m_dow = pd.DataFrame
        forecasts_m_dow = df.filter(['departure_date','booking_date','dow_forecast'], axis=1)
        
        return MASE_dow, forecasts_m_dow

    
    def mul_model_dop(validationDataFileName,trainingDataFileName):
        tdt = trainingDataFileName
        vdt = validationDataFileName
        """ days prior """
        #trainning data  
        # Multiplicative model
        #trainning data  
        # days prior = depature - book
        tdt['days prior'] = (pd.to_datetime(tdt['departure_date']) - pd.to_datetime(tdt['booking_date'])).dt.days
        
        #total demand of tickets for each departure day
        tdt['final'] = tdt.loc[tdt['days prior'] == 0,'cum_bookings']
        final = tdt.loc[tdt['days prior'] == 0, ['departure_date', 'final']]
        tdt = tdt.merge(final, left_on =['departure_date'], right_on = ['departure_date'])
        tdt = tdt.drop('final_x', axis = 1)
        tdt = tdt.rename(columns = {'final_y': 'final_demanding'})
        
        #booking rate
        tdt['booking_rate'] = tdt['cum_bookings']/tdt['final_demanding']
        
        #avg booking rate by same days prior
        avg_booking = tdt.groupby(['days prior']).mean()
        avg_booking['day'] = pd.DataFrame(avg_booking.index)
        avg_booking = avg_booking.iloc[0:29]
        avg_booking = avg_booking.drop(['cum_bookings', 'final_demanding'], axis=1)
        
        #validation data  
        vdt['days prior'] = (pd.to_datetime(vdt['departure_date']) - pd.to_datetime(vdt['booking_date'])).dt.days
        new1 = avg_booking.merge(vdt, how = 'left' , left_on = ['day'], right_on = ['days prior'])
        
        # error: actual demand - forecast
        new1['forecast'] = new1['cum_bookings']/new1['booking_rate']
        new1['error_multiplicative'] =abs(new1['final_demand'] - (new1['cum_bookings']/new1['booking_rate']))
        new1['error_naive'] = abs(new1['final_demand'] - new1['naive_forecast'])
        new1 = new1.iloc[7:]
        
        # MASE for multiplicative model
        MASE_dop = new1['error_multiplicative'].sum() / new1['error_naive'].sum()
        
        #forecast of multiplicative model with days prior
        forecasts_m_dop = pd.DataFrame
        forecasts_m_dop = new1.filter(['departure_date','booking_date','forecast'], axis=1)
    
        return MASE_dop, forecasts_m_dop


    def add_model(validationDataFileName,trainingDataFileName):
        """ Additive model"""
        tdt = trainingDataFileName
        vdt = validationDataFileName
        #trainning data  
        # days prior = depature - book 
        tdt['days prior'] = (pd.to_datetime(tdt['departure_date']) - pd.to_datetime(tdt['booking_date'])).dt.days
    
        #total demand of tickets for each departure day
        #tdt['final'] = tdt.loc[tdt['days prior'] == 0,'cum_bookings']
        final = tdt.loc[tdt['days prior'] == 0, ['departure_date', 'cum_bookings']]
        tdt = tdt.merge(final, left_on =['departure_date'], right_on = ['departure_date'])
        #tdt = tdt.drop('final_x', axis = 1)
        tdt = tdt.rename(columns = {'cum_bookings_y': 'final_demanding','cum_bookings_x': 'cum_bookings'})
        
        #remaining tickets for each departure day
        tdt['remaining'] = tdt['final_demanding'] - tdt['cum_bookings']
                
        #average demand tickets for same prior days for 28 days 
        avg = tdt.groupby(['days prior']).mean()
        avg['d'] = pd.DataFrame(avg.index)
        avg = avg.iloc[0:29]
        avg= avg.drop(['cum_bookings', 'final_demanding'], axis=1)
        
        #validation data  
        vdt['days prior'] = (pd.to_datetime(vdt['departure_date']) - pd.to_datetime(vdt['booking_date'])).dt.days
        
        new = vdt.merge(avg, how = 'left' , left_on = ['days prior'], right_on = ['d'])
        
        new= new.loc[new["days prior"] != 0]
        
        #error: actual demand - forecast
        new['forecast']= new['remaining']+ new['cum_bookings']
        new['error_additive'] =abs(new['final_demand'] - new['forecast'])
        new['error_naive'] = abs(new['final_demand'] - new['naive_forecast'])
        
        # MASE for additive model
        MASE_add = new['error_additive'].sum() / new['error_naive'].sum()
        forecasts_add = new.filter(['departure_date','booking_date','forecast', 'final_demand'], axis=1)
        return MASE_add,forecasts_add
         
         
    def add_model_dow(validationDataFileName,trainingDataFileName):
        
        tdt = trainingDataFileName
        vdt = validationDataFileName
        
        #trainning data  
        # days prior = depature - book
        tdt['days prior'] = (pd.to_datetime(tdt['departure_date']) - pd.to_datetime(tdt['booking_date'])).dt.days
        tdeparture_date = pd.to_datetime(tdt['departure_date'])
        tdt['day_of_week']= tdeparture_date.dt.weekday_name
            
        #total demand of tickets for each departure day
        final = tdt.loc[tdt['days prior'] == 0, ['departure_date', 'cum_bookings']]
        tdt = tdt.merge(final, left_on =['departure_date'], right_on = ['departure_date'])
        tdt = tdt.rename(columns = {'cum_bookings_y': 'final_demanding','cum_bookings_x': 'cum_bookings'})
            
        #remaining tickets for each departure day
        tdt['remaining'] = tdt['final_demanding'] - tdt['cum_bookings']
        
        #average demand tickets for same prior days for 28 days 
        avg = tdt.groupby(['days prior','day_of_week']).mean().reset_index()
        avg = avg.iloc[0:203]
        avg= avg.drop(['cum_bookings', 'final_demanding'], axis=1)
            
        #validation data  
        vdt['days prior'] = (pd.to_datetime(vdt['departure_date']) - pd.to_datetime(vdt['booking_date'])).dt.days
        Vdeparture_date = pd.to_datetime(vdt['departure_date'])
        vdt['day_of_week']= Vdeparture_date.dt.weekday_name
        new = vdt.merge(avg, how = 'left' , left_on = ['days prior', 'day_of_week'], right_on = ['days prior','day_of_week'])
        
        new= new.loc[new["days prior"] != 0]
        
        #error: actual demand - forecast
        new['forecast']= new['remaining'] + new['cum_bookings']
        new['error_mix'] =abs(new['final_demand'] - new['forecast'])
        new['error_naive'] = abs(new['final_demand'] - new['naive_forecast'])
    
        # MASE for mix model
        MASE_add_dow = new['error_mix'].sum() / new['error_naive'].sum()
        forecasts_add_dow = new.filter(['departure_date','booking_date','forecast'], axis=1)
        return MASE_add_dow,forecasts_add_dow
               
    
    MASE_dow, forecasts_m_dow = mul_model_dow(vdt,tdt)
    MASE_dop, forecasts_m_dop = mul_model_dop(vdt,tdt)
    MASE_add, forecasts_add = add_model(vdt,tdt)
    MASE_add_dow, forecasts_add_dow = add_model_dow(vdt,tdt)

    
    def Min(a,b,c,d):
        Min = a
        if b < Min:
            Min = b
        if c < Min:
            Min = c
        if d < Min:
            Min = d
        return(Min)
        
    result_min = Min(MASE_dow,MASE_dop,MASE_add,MASE_add_dow)
    
    if result_min is MASE_dow:
        result_df = forecasts_m_dow
    elif result_min is MASE_dop:
        result_df = forecasts_m_dop
    elif result_min is MASE_add:
        result_df = forecasts_add
    elif result_min is MASE_add_dow:
        result_df = forecasts_add_dow
        
    return(result_min, result_df)
    
def main():
    print(airlineForecast("airline_booking_validationData.csv", "airline_booking_trainingData.csv"))
    
main() 
