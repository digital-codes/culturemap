<script setup lang="ts">
import { ref } from 'vue';
import { useI18n } from 'vue-i18n';
const { t } = useI18n() 

import { JsonForms } from '@jsonforms/vue';
import { vanillaRenderers } from '@jsonforms/vue-vanilla';

const renderers = Object.freeze([
    ...vanillaRenderers,
    // here you can add custom renderers
])

const data1 = ref({
    number: 5,
})

const schema1 = {
    properties: {
        number: {
            type: 'number',
        },
    },
}
const uischema1 = {
    type: 'VerticalLayout',
    elements: [
        {
            type: 'Control',
            scope: '#/properties/number',
        },
    ],
}

const loggedIn = ref(false);

const token = ref("");

const data = ref({
    passkey: "",
    vegetables: false,
    kindOfVegetables: "All",
    vitaminDeficiency: "None"
})

const schema = {
    "type": "object",
    "properties": {
        "passkey": {
            "type": "string"
        },
        "vegetables": {
            "type": "boolean"
        },
        "kindOfVegetables": {
            "type": "string",
            "enum": [
                "All",
                "Some",
                "Only potatoes"
            ]
        },
        "vitaminDeficiency": {
            "type": "string",
            "enum": [
                "None",
                "Vitamin A",
                "Vitamin B",
                "Vitamin C"
            ]
        }
    }
}

const uischema = {
    "type": "VerticalLayout",
    "elements": [
        {
            "type": "Control",
            "label": t('message.passkey'),
            "scope": "#/properties/passkey"
        },
        {
            "type": "Group",
            "rule": {
                "effect": "HIDE",
                "condition": {
                    "scope": "#/properties/vegetables",
                    "schema": {
                        "const": false
                    }
                }
            },
            "elements": [
                {
                    "type": "Control",
                    "label": t('message.eatsVegetables'),
                    "scope": "#/properties/vegetables"
                },
                {
                    "type": "Control",
                    "label": t('message.kindOfVegetables'),
                    "scope": "#/properties/kindOfVegetables",
                    "rule": {
                        "effect": "HIDE",
                        "condition": {
                            "scope": "#/properties/vegetables",
                            "schema": {
                                "const": false
                            }
                        }
                    }
                },
                {
                    "type": "Control",
                    "label": t('message.vitaminDeficiency'),
                    "scope": "#/properties/vitaminDeficiency",
                    "rule": {
                        "effect": "SHOW",
                        "condition": {
                            "scope": "#"
                        }
                    }
                }
            ]
        }
    ]
}


const onChange = async (event) => {
    data.value = event.data;
    console.log("New data: ", data.value);
    if (!loggedIn.value) {
        console.log("Need loging first, ignoring changes.");
        token.value = "";
        const passkey = data.value.passkey;
        if (passkey === "") {
            return; // ignore
        }
        try {
            const r = await fetch('/php/llamaLogin.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ password: passkey, username:"any" })
            })
            if (r.status === 200) {
                const params = await r.json();
                token.value = params.token;
                loggedIn.value = true;
                data.value.vegetables = true;
                return; 
            } 
            console.error('Error checking passkey:', r.statusText);
            alert("An error occurred while checking the passkey. Please try again later.");
            return;
        }
        catch (error) {
            console.error('Error checking passkey:', error);
            alert("An error occurred while checking the passkey. Please try again later.");
            return;
        }
    }
}

</script>

<template>
    <div class="about">
        <p>{{ $t('message.about') }}</p>
        <json-forms :data="data" :schema="schema" :uischema="uischema" :renderers="renderers" @change="onChange" />

    </div>
</template>
