/** @file
 *  pulsar-agent: the shared globals (agent_sigint, agent_completion_worker; declared in pulsar_agent_internal.h). */
#include "pulsar_agent_internal.h"



volatile sig_atomic_t agent_sigint;
agent_worker *agent_completion_worker;

