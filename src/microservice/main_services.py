'''
    Microservice services launcher

    This module provides functionality to launch different microservices.

    Usage examples:
    python3 -m src.microservice.main_services # launch all services
    python3 -m src.microservice.main_services --service-name all # launch all services
    python3 -m src.microservice.main_services --service-name offline # launch offline recommendations service
    python3 -m src.microservice.main_services --service-name events # launch events service
    python3 -m src.microservice.main_services --service-name online # launch online recommendations service
    python3 -m src.microservice.main_services --service-name final # launch final recommendations service
'''

# ---------- Imports ---------- #
import os
from dotenv import load_dotenv
import argparse
import uvicorn
from multiprocessing import Process

from src.logging_setup import setup_logging

# ---------- Logging setup ---------- #
logger = setup_logging('main_services')

# ---------- Load environment variables ---------- #
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, '.env'))

main_app_port = int(os.getenv('MAIN_APP_PORT', 8000))
offline_recs_service_port = int(os.getenv('OFFLINE_RECS_SERVICE_PORT', 8001))
events_service_port = int(os.getenv('EVENTS_SERVICE_PORT', 8002))
online_recs_service_port = int(os.getenv('ONLINE_RECS_SERVICE_PORT', 8003))

# ---------- Service runner functions ---------- #
def run_offline_service():
    from src.microservice import offline_recs
    uvicorn.run(app=offline_recs.app, host='127.0.0.1', port=offline_recs_service_port)

def run_events_service():
    from src.microservice import events
    uvicorn.run(app=events.app, host='127.0.0.1', port=events_service_port)

def run_online_service():
    from src.microservice import online_recs
    uvicorn.run(app=online_recs.app, host='127.0.0.1', port=online_recs_service_port)

def run_final_service():
    from src.microservice import final_recs
    uvicorn.run(app=final_recs.app, host='127.0.0.1', port=main_app_port)

# ---------- Main entry point ---------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Starting Services Main Pipeline')
    parser.add_argument('--service-name', choices=['all', 'offline', 'events', 'online', 'final'], default='all')
    args = parser.parse_args()

    if args.service_name == 'all':
        # Starting all services using multiprocessing
        logger.info('Starting all services')
        
        processes = [
            Process(target=run_offline_service, name='offline_recs'),
            Process(target=run_events_service, name='events'),
            Process(target=run_online_service, name='online_recs'),
            Process(target=run_final_service, name='final_recs'),
        ]
        
        # Start all processes
        for p in processes:
            p.start()
            logger.info(f'Started {p.name} service')
        
        # Wait for all processes
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logger.info('Shutting down all services...')
            for p in processes:
                p.terminate()
    
    elif args.service_name == 'offline':
        logger.info('Starting offline recommendations service')
        run_offline_service()
    
    elif args.service_name == 'events':
        logger.info('Starting events service')
        run_events_service()

    elif args.service_name == 'online':
        logger.info('Starting online recommendations service')
        run_online_service()

    elif args.service_name == 'final':
        logger.info('Starting final recommendations service')
        run_final_service()

    else:
        logger.error(f'Invalid service name: {args.service_name}')
        exit(1)



