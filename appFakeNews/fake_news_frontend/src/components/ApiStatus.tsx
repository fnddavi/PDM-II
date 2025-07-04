// src/components/ApiStatus.tsx
import React, { useEffect, useState } from 'react';
import { API_BASE_URL } from '../api';
import { Box, Typography, Paper, CircularProgress, Alert } from '@mui/material'; // Importar componentes MUI
import CheckCircleIcon from '@mui/icons-material/CheckCircle'; // Ícone de sucesso
import ErrorIcon from '@mui/icons-material/Error'; // Ícone de erro

interface ApiStatusData {
    modelo: string;
    vetorizador: string;
    treinado: boolean;
}

const ApiStatus: React.FC = () => {
    const [status, setStatus] = useState<ApiStatusData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(true);

    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/status`);
                if (!response.ok) {
                    throw new Error('Erro ao buscar o status da API.');
                }
                const data: ApiStatusData = await response.json();
                setStatus(data);
            } catch (err: unknown) {
                if (err instanceof Error) {
                    setError(err.message);
                } else {
                    setError('Erro desconhecido ao buscar o status da API.');
                }
            } finally {
                setLoading(false);
            }
        };

        fetchStatus();
    }, []);

    if (loading) {
        return (
            <Box display="flex" alignItems="center" justifyContent="center" sx={{ p: 2 }}>
                <CircularProgress size={24} sx={{ mr: 2 }} />
                <Typography variant="body1">Verificando status da API...</Typography>
            </Box>
        );
    }

    if (error) {
        return (
            <Alert severity="error" icon={<ErrorIcon fontSize="inherit" />}>
                <Typography variant="body1">Erro ao conectar à API: {error}</Typography>
            </Alert>
        );
    }

    return (
        <Paper elevation={1} sx={{ p: 3, bgcolor: status?.treinado ? '#e8f5e9' : '#fff3e0', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box>
                <Typography variant="h6" component="h3" gutterBottom>
                    Status da API
                </Typography>
                {status ? (
                    <>
                        <Typography variant="body2"><strong>Modelo:</strong> {status.modelo}</Typography>
                        <Typography variant="body2"><strong>Vetorizador:</strong> {status.vetorizador}</Typography>
                        <Typography variant="body2"><strong>Treinado:</strong> {status.treinado ? 'Sim' : 'Não'}</Typography>
                    </>
                ) : (
                    <Typography variant="body2">Não foi possível obter o status da API.</Typography>
                )}
            </Box>
            {status?.treinado ? (
                <CheckCircleIcon color="success" sx={{ fontSize: 40 }} />
            ) : (
                <ErrorIcon color="warning" sx={{ fontSize: 40 }} />
            )}
        </Paper>
    );
};

export default ApiStatus;
