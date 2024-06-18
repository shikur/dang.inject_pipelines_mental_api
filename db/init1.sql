
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'session_db') THEN
        CREATE DATABASE session_db;
        CREATE USER postgres_user  WITH PASSWORD 'postgres_password';
        GRANT ALL PRIVILEGES ON DATABASE session_db to postgres_user;
    END IF;
END
$$;

-- Connect to the newly created database
\c session_db;

-- Attempt to connect to the target database (this line is pseudocode and must be executed outside of SQL)
\c session_db;

-- Check if the table exists, and create it if it does not
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_tables 
                   WHERE  schemaname = 'public' 
                   AND    tablename  = 'session_video') THEN
        CREATE TABLE public.session_video (
            id SERIAL PRIMARY KEY,
            angry FLOAT,
            disgust FLOAT,
            fear FLOAT,
            happy FLOAT,
            neutral FLOAT,
            sad FLOAT,
            surprise FLOAT,
            data VARCHAR(255)
        );
    END IF;
END $$;

-- Insert some sample data
INSERT INTO session_video (angry, disgust, fear, happy, neutral, sad, surprise, data ) VALUES (0.2, 0.3, 0.5, 0.4, 0.6,0.7,0.8 ,'data');
