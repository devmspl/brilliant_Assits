import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCvpDx_01BS5T0r-knle2sglcAoyJ1RBDk')

model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

DATA_FILE = 'data/dataset.csv'

# Supported categories with descriptions
SUPPORTED_CATEGORIES = {
    'social networking': 'Social media platforms and their metrics',
    'real estate': 'Property prices and market trends',
    'financial': 'Stock market and financial data',
    'software/it': 'Technology sector metrics',
    'e-commerce': 'Online retail statistics',
    'tourism': 'Travel and hospitality data',
    'market analysis': 'Industry market trends',
    'health': 'Healthcare statistics',
    'service provider': 'Service industry metrics',
    'population': 'Demographic data',
    'pollution': 'Environmental data'
}

def get_historical_data(category, years=5):
    """Get historical data for a category using Gemini AI"""
    try:
        prompt = f"""Generate realistic historical data for the '{category}' category covering the past {years} years.
        Return the data in strict JSON format with this structure:
        {{
            "description": "Brief description of the data",
            "units": "Measurement units",
            "data": [
                {{
                    "year": 2023,
                    "value": 123.45,
                    "notes": "Any relevant notes"
                }},
                // more years...
            ]
        }}
        
        Requirements:
        1. Include data for each of the past {years} years
        2. Values should reflect realistic trends
        3. Include variation that shows meaningful patterns
        4. Return ONLY the JSON object with no additional text
        5. Ensure all values are numbers where applicable"""
        
        logger.info(f"Requesting historical data for: {category}")
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean the response
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
        
        logger.info(f"Cleaned response: {response_text}")
        return json.loads(response_text)
        
    except Exception as e:
        logger.error(f"Error getting historical data: {str(e)}")
        return None

def analyze_data(category, data):
    """Get AI analysis of the collected data"""
    try:
        prompt = f"""Analyze this {category} data and provide insights:
        {data}
        
        Provide:
        1. Key trends observed
        2. Notable patterns or anomalies
        3. Potential causes for the trends
        4. Future projections based on the data
        5. Any recommendations
        
        Format your response in clear paragraphs with bullet points where appropriate."""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        return "Analysis unavailable at this time."

def create_trend_chart(data):
    """Create a line chart of the historical trends"""
    years = [str(item['year']) for item in data['data']]
    values = [item['value'] for item in data['data']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(years, values, marker='o')
    plt.title(f"{data['description']} ({data['units']})")
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.grid(True)
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode('utf8')

@app.route('/')
def index():
    return render_template('index.html', categories=SUPPORTED_CATEGORIES)

@app.route('/collect', methods=['GET', 'POST'])
def collect():
    if request.method == 'POST':
        category = request.form.get('category')
        years = int(request.form.get('years', 5))
        
        if category not in SUPPORTED_CATEGORIES:
            return render_template('collect.html', 
                                categories=SUPPORTED_CATEGORIES,
                                error="Please select a valid category")
        
        historical_data = get_historical_data(category, years)
        
        if not historical_data:
            return render_template('collect.html',
                                categories=SUPPORTED_CATEGORIES,
                                error="Failed to collect data. Please try again.")
        
        # Save the data
        df = pd.DataFrame(historical_data['data'])
        df['category'] = category
        df['description'] = historical_data['description']
        df['units'] = historical_data['units']
        
        try:
            existing_df = pd.read_csv(DATA_FILE)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass
            
        df.to_csv(DATA_FILE, index=False)
        
        return redirect(url_for('analyze', category=category))
    
    return render_template('collect.html', categories=SUPPORTED_CATEGORIES)

@app.route('/analyze')
def analyze():
    category = request.args.get('category')
    
    try:
        df = pd.read_csv(DATA_FILE)
        category_data = df[df['category'] == category]
        
        if category_data.empty:
            return render_template('analyze.html',
                                categories=SUPPORTED_CATEGORIES,
                                message="No data available for this category.")
        
        # Prepare data for visualization
        data = {
            'description': category_data['description'].iloc[0],
            'units': category_data['units'].iloc[0],
            'data': category_data[['year', 'value']].to_dict('records')
        }
        
        # Create chart
        chart_image = create_trend_chart(data)
        
        # Get AI analysis
        analysis = analyze_data(category, data)
        
        return render_template('analyze.html',
                            categories=SUPPORTED_CATEGORIES,
                            category=category,
                            chart_image=chart_image,
                            data=data,
                            analysis=analysis)
        
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        return render_template('analyze.html',
                            categories=SUPPORTED_CATEGORIES,
                            message="Error processing data. Please try again.")

if __name__ == '__main__':
    app.run(debug=True, port=5001)