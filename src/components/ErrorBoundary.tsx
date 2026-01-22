import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
    children: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
}

class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null,
    };

    public static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('Uncaught error:', error, errorInfo);
    }

    private handleReload = () => {
        window.location.reload();
    };

    public render() {
        if (this.state.hasError) {
            return (
                <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
                    <div className="bg-gray-800 rounded-lg p-8 max-w-md w-full text-center">
                        <AlertTriangle className="mx-auto h-16 w-16 text-yellow-500 mb-4" />
                        <h1 className="text-2xl font-bold text-white mb-2">Something went wrong</h1>
                        <p className="text-gray-400 mb-6">
                            An unexpected error occurred. This might be due to a problem loading the image analysis models.
                        </p>
                        {this.state.error && (
                            <details className="mb-6 text-left">
                                <summary className="text-gray-500 cursor-pointer hover:text-gray-400">
                                    Technical details
                                </summary>
                                <pre className="mt-2 p-3 bg-gray-900 rounded text-red-400 text-sm overflow-auto max-h-32">
                                    {this.state.error.message}
                                </pre>
                            </details>
                        )}
                        <button
                            onClick={this.handleReload}
                            className="flex items-center justify-center gap-2 w-full bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-md transition-colors duration-200"
                        >
                            <RefreshCw size={18} />
                            Reload Page
                        </button>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
