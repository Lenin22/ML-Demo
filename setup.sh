mkdir -p ~/.streamlit/
echo "[general]
email = \"sirazievlenar@mail.ru\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml