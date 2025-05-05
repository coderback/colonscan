FROM nginx:stable-alpine

# Copy in your custom nginx.conf
COPY nginx.conf /etc/nginx/nginx.conf

# Expose HTTP
EXPOSE 80

# Run Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
