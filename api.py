"""
Simple Flask API for the Knowledge Graph Pipeline

Provides REST endpoints for:
- Text processing
- Graph retrieval
- Graph statistics
"""

from flask import Flask, request, jsonify
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import TextToKnowledgeGraphPipeline

app = Flask(__name__)

# Initialize pipeline (global, loaded once)
pipeline = None


def init_pipeline():
    """Initialize the pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = TextToKnowledgeGraphPipeline()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Text to Knowledge Graph API"
    })


@app.route('/process', methods=['POST'])
def process_text():
    """
    Process text and return knowledge graph.
    
    Request JSON:
    {
        "text": "Educational text here..."
    }
    
    Response:
    {
        "status": "success",
        "graph": {...}
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                "status": "error",
                "message": "No text provided"
            }), 400
        
        init_pipeline()
        
        # Run pipeline (simplified - doesn't save to file)
        stage1 = pipeline.stage_1_information_extraction(text)
        stage2 = pipeline.stage_2_wsd_and_normalization(stage1, text)
        stage3 = pipeline.stage_3_ontology_resolution(stage1, stage2)
        graph = pipeline.stage_4_graph_construction(stage1, stage2, stage3)
        
        return jsonify({
            "status": "success",
            "graph": graph.to_dict()
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/graph/stats', methods=['GET'])
def graph_stats():
    """Get statistics about the current graph."""
    init_pipeline()
    return jsonify(pipeline.graph_constructor.get_statistics())


if __name__ == '__main__':
    init_pipeline()
    app.run(debug=True, port=5000)