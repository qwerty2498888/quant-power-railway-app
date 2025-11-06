import dash
from dash import html, dcc
import os
import sys

print('=== MAX POWER DIAGNOSTICS ===')
print('Python:', sys.version)
print('Workdir:', os.getcwd())
print('Files:', [f for f in os.listdir('.') if f.endswith('.py') or f.endswith('.txt')])

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("🎉 Max Power - IT WORKS!", style={
        'textAlign': 'center', 
        'color': '#00ffcc',
        'marginTop': '100px'
    }),
    html.P("Если вы видите эту страницу, ваш сайт успешно работает на Railway!", style={
        'textAlign': 'center', 
        'color': 'white',
        'fontSize': '20px'
    }),
], style={
    'backgroundColor': '#1e1e1e', 
    'minHeight': '100vh',
    'padding': '50px'
})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f'🚀 STARTING SERVER ON PORT {port}')
    app.run(host='0.0.0.0', port=port, debug=False)
